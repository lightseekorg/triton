#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace py = nanobind;

#define PY_ARRAY_UNIQUE_SYMBOL triton_interpreter_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace {

struct triton_half {
  uint16_t value;
};

enum class MemSemantic { ACQUIRE_RELEASE, ACQUIRE, RELEASE, RELAXED };

std::mutex atomic_op_guard;

template <typename T>
constexpr bool is_reinterpret_cast_to_atomic_safe =
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_copyable_v<std::atomic<T>> &&
    std::is_standard_layout_v<T> && std::is_standard_layout_v<std::atomic<T>> &&
    sizeof(T) == sizeof(std::atomic<T>) &&
    alignof(T) == alignof(std::atomic<T>);

enum class RMWOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };

std::map<MemSemantic, std::memory_order> mem_semantic_map = {
    {MemSemantic::ACQUIRE_RELEASE, std::memory_order_acq_rel},
    {MemSemantic::ACQUIRE, std::memory_order_acquire},
    {MemSemantic::RELEASE, std::memory_order_release},
    {MemSemantic::RELAXED, std::memory_order_relaxed},
};

template <bool is_min, typename T>
T atomic_cmp(T *ptr, T val, std::memory_order order) {
  auto cmp = [](T old, T val) {
    if constexpr (is_min) {
      return old > val;
    } else {
      return old < val;
    }
  };

  T old_val;
  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    std::atomic<T> *atomic_ptr = reinterpret_cast<std::atomic<T> *>(ptr);
    old_val = atomic_ptr->load(order);
    while (cmp(old_val, val)) {
      if (atomic_ptr->compare_exchange_weak(old_val, val, order, order)) {
        break;
      }
    }
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_val = *ptr;
    if (cmp(old_val, val)) {
      *ptr = val;
    }
  }
  return old_val;
}

template <typename T> T atomic_fadd(T *loc, T value, std::memory_order order) {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type");
  T old_value;

  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    T new_value;
    std::atomic<T> *atomic_loc = reinterpret_cast<std::atomic<T> *>(loc);
    old_value = atomic_loc->load(order);
    do {
      new_value = old_value + value;
    } while (
        !atomic_loc->compare_exchange_weak(old_value, new_value, order, order));
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_value = *loc;
    *loc = old_value + value;
  }

  return old_value;
}

/** Create a value of type `To` from the bits of `from`.
 *
 * similar to `std::bit_cast` but compatible with C++17,
 * should perform similar to `*reinterpret_cast<To*>(&from)`
 * or through punning without expecting any undefined behaviors.
 *
 * Note: taken from
 * https://github.com/numpy/numpy/blob/70fde29fdd4d8fcc6098df7ef8a34c84844e347f/numpy/_core/src/common/utils.hpp#L32
 * with simplification.
 */
template <typename To, typename From>
inline To BitCast(const From &from) noexcept {
  static_assert(sizeof(To) == sizeof(From),
                "both data types must have the same size");

  static_assert(std::is_trivially_copyable_v<To> &&
                    std::is_trivially_copyable_v<From>,
                "both data types must be trivially copyable");

  To to;
  memcpy(&to, &from, sizeof(from));
  return to;
}

// Taken from
// https://github.com/numpy/numpy/blob/70fde29fdd4d8fcc6098df7ef8a34c84844e347f/numpy/_core/src/common/half_private.hpp#L14
template <bool gen_overflow = true, bool gen_underflow = true,
          bool round_even = true>
inline uint16_t FromFloatBits(uint32_t f) {
  uint32_t f_exp, f_sig;
  uint16_t h_sgn, h_exp, h_sig;

  h_sgn = (uint16_t)((f & 0x80000000u) >> 16);
  f_exp = (f & 0x7f800000u);

  /* Exponent overflow/NaN converts to signed inf/NaN */
  if (f_exp >= 0x47800000u) {
    if (f_exp == 0x7f800000u) {
      /* Inf or NaN */
      f_sig = (f & 0x007fffffu);
      if (f_sig != 0) {
        /* NaN - propagate the flag in the significand... */
        uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
        /* ...but make sure it stays a NaN */
        if (ret == 0x7c00u) {
          ret++;
        }
        return h_sgn + ret;
      } else {
        /* signed inf */
        return (uint16_t)(h_sgn + 0x7c00u);
      }
    } else {
      if constexpr (gen_overflow) {
        // FloatStatus::RaiseOverflow();
        throw std::overflow_error("overflow to signed inf");
      }
      return (uint16_t)(h_sgn + 0x7c00u);
    }
  }

  /* Exponent underflow converts to a subnormal half or signed zero */
  if (f_exp <= 0x38000000u) {
    /*
     * Signed zeros, subnormal floats, and floats with small
     * exponents all convert to signed zero half-floats.
     */
    if (f_exp < 0x33000000u) {
      if constexpr (gen_underflow) {
        /* If f != 0, it underflowed to 0 */
        if ((f & 0x7fffffff) != 0) {
          // FloatStatus::RaiseUnderflow();
          throw std::underflow_error("");
        }
      }
      return h_sgn;
    }
    /* Make the subnormal significand */
    f_exp >>= 23;
    f_sig = (0x00800000u + (f & 0x007fffffu));
    if constexpr (gen_underflow) {
      /* If it's not exactly represented, it underflowed */
      if ((f_sig & (((uint32_t)1 << (126 - f_exp)) - 1)) != 0) {
        // FloatStatus::RaiseUnderflow();
        throw std::underflow_error("");
      }
    }
    /*
     * Usually the significand is shifted by 13. For subnormals an
     * additional shift needs to occur. This shift is one for the largest
     * exponent giving a subnormal `f_exp = 0x38000000 >> 23 = 112`, which
     * offsets the new first bit. At most the shift can be 1+10 bits.
     */
    f_sig >>= (113 - f_exp);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    if constexpr (round_even) {
      /*
       * If the last bit in the half significand is 0 (already even), and
       * the remaining bit pattern is 1000...0, then we do not add one
       * to the bit after the half significand. However, the (113 - f_exp)
       * shift can lose up to 11 bits, so the || checks them in the original.
       * In all other cases, we can just add one.
       */
      if (((f_sig & 0x00003fffu) != 0x00001000u) || (f & 0x000007ffu)) {
        f_sig += 0x00001000u;
      }
    } else {
      f_sig += 0x00001000u;
    }
    h_sig = (uint16_t)(f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp from zero to one and h_sig will be zero.
     * This is the correct result.
     */
    return (uint16_t)(h_sgn + h_sig);
  }

  /* Regular case with no overflow or underflow */
  h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
  /* Handle rounding by adding 1 to the bit beyond half precision */
  f_sig = (f & 0x007fffffu);
  if constexpr (round_even) {
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig & 0x00003fffu) != 0x00001000u) {
      f_sig += 0x00001000u;
    }
  } else {
    f_sig += 0x00001000u;
  }
  h_sig = (uint16_t)(f_sig >> 13);
  /*
   * If the rounding causes a bit to spill into h_exp, it will
   * increment h_exp by one and h_sig will be zero.  This is the
   * correct result.  h_exp may increment to 15, at greatest, in
   * which case the result overflows to a signed inf.
   */
  if constexpr (gen_overflow) {
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
      // FloatStatus::RaiseOverflow();
      throw std::overflow_error("");
    }
    return h_sgn + h_sig;
  } else {
    return h_sgn + h_exp + h_sig;
  }
}

// Taken from
// https://github.com/numpy/numpy/blob/70fde29fdd4d8fcc6098df7ef8a34c84844e347f/numpy/_core/src/common/half_private.hpp#L269
constexpr uint32_t ToFloatBits(uint16_t h) {
  uint16_t h_exp = (h & 0x7c00u);
  uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;
  switch (h_exp) {
  case 0x0000u: { // 0 or subnormal
    uint16_t h_sig = (h & 0x03ffu);
    // Signed zero
    if (h_sig == 0) {
      return f_sgn;
    }
    // Subnormal
    h_sig <<= 1;
    while ((h_sig & 0x0400u) == 0) {
      h_sig <<= 1;
      h_exp++;
    }
    uint32_t f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
    uint32_t f_sig = ((uint32_t)(h_sig & 0x03ffu)) << 13;
    return f_sgn + f_exp + f_sig;
  }
  case 0x7c00u: // inf or NaN
    // All-ones exponent and a copy of the significand
    return f_sgn + 0x7f800000u + (((uint32_t)(h & 0x03ffu)) << 13);
  default: // normalized
    // Just need to adjust the exponent and shift
    return f_sgn + (((uint32_t)(h & 0x7fffu) + 0x1c000u) << 13);
  }
}

triton_half npy_float_to_half(float f) {
  return {FromFloatBits(BitCast<uint32_t>(f))};
}

float npy_half_to_float(triton_half h) {
  return BitCast<float>(ToFloatBits(h.value));
}

template <>
triton_half atomic_fadd<triton_half>(triton_half *loc, triton_half value,
                                     std::memory_order order) {
  triton_half old_value;

  const std::lock_guard<std::mutex> lock(atomic_op_guard);
  old_value = *loc;
  *loc = npy_float_to_half(npy_half_to_float(old_value) +
                           npy_half_to_float(value));

  return old_value;
}

class AtomicOp {
public:
  AtomicOp(const uint64_t *ptr, size_t numel, std::memory_order order)
      : ptr(ptr), numel(numel), order(order) {}

  void apply() {
    for (size_t i = 0; i < numel; ++i) {
      applyAt(reinterpret_cast<void *>(ptr[i]), i);
    }
  }

  virtual ~AtomicOp() = default;

protected:
  virtual void applyAt(void *, size_t i) = 0;

  const uint64_t *ptr;
  size_t numel;
  std::memory_order order;
};

template <typename DType> class AtomicRMWOpBase : public AtomicOp {
public:
  AtomicRMWOpBase(const uint64_t *ptr, const void *val, void *ret,
                  const bool *mask, size_t numel, std::memory_order order)
      : AtomicOp(ptr, numel, order), val(val), ret(ret), mask(mask) {}

protected:
  void applyAt(void *loc, size_t i) override final {
    if (mask[i]) {
      DType *ptr = static_cast<DType *>(loc);
      *(static_cast<DType *>(ret) + i) =
          applyAtMasked(ptr, *(static_cast<const DType *>(val) + i), order);
    }
  }

  virtual DType applyAtMasked(DType *loc, const DType value,
                              std::memory_order order) = 0;

  const void *val;
  void *ret;
  const bool *mask;
};

template <typename DType, RMWOp Op, typename = void>
class AtomicRMWOp : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::ADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_add_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc + value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::FADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_fadd(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::AND>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_and_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc & value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::OR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_or_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc | value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XOR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_xor_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc ^ value;
    }
    return old_val;
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MAX || Op == RMWOp::UMAX>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_cmp</*is_min=*/false>(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MIN || Op == RMWOp::UMIN>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_cmp</*is_min=*/true>(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XCHG>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = atomic_loc->exchange(value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = value;
    }
    return old_val;
  }
};

template <typename T>
void atomic_compare_exchange_strong(void *loc, void *expected,
                                    const void *desired, size_t i,
                                    std::memory_order order) {
  T desired_val = *(static_cast<const T *>(desired) + i);
  T *expected_uint = static_cast<T *>(expected) + i;

  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    std::atomic<T> *atomic_loc = reinterpret_cast<std::atomic<T> *>(loc);
    atomic_loc->compare_exchange_strong(*expected_uint, desired_val, order,
                                        order);
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    T *atomic_loc = static_cast<T *>(loc);
    if (*atomic_loc == *expected_uint) {
      *atomic_loc = desired_val;
    } else {
      *expected_uint = *atomic_loc;
    }
  }
}

class AtomicCASOp : public AtomicOp {
public:
  AtomicCASOp(const uint64_t *ptr, void *expected, const void *desired,
              size_t itemsize, size_t numel, std::memory_order order)
      : AtomicOp(ptr, numel, order), expected(expected), desired(desired),
        itemsize(itemsize) {}

protected:
  void applyAt(void *loc, size_t i) override {
    // Atomic operations perform bitwise comparison, so it's safe to
    // use number of bytes (itemsize) to determine the type of pointers
    if (itemsize == 1) {
      atomic_compare_exchange_strong<uint8_t>(loc, expected, desired, i, order);
    } else if (itemsize == 2) {
      atomic_compare_exchange_strong<uint16_t>(loc, expected, desired, i,
                                               order);
    } else if (itemsize == 4) {
      atomic_compare_exchange_strong<uint32_t>(loc, expected, desired, i,
                                               order);
    } else if (itemsize == 8) {
      atomic_compare_exchange_strong<uint64_t>(loc, expected, desired, i,
                                               order);
    } else {
      throw std::invalid_argument("Invalid byte size");
    }
  }

private:
  void *expected;
  const void *desired;
  size_t itemsize;
};

PyArrayObject *require_ndarray(const py::object &obj) {
  if (!PyArray_Check(obj.ptr())) {
    throw std::invalid_argument("expected a numpy ndarray");
  }
  return reinterpret_cast<PyArrayObject *>(obj.ptr());
}

bool numpy_dtypes_equal(const py::object &a, const py::object &b) {
  py::module_ np = py::module_::import_("numpy");
  py::object da = np.attr("dtype")(a);
  py::object db = np.attr("dtype")(b);
  int r = PyObject_RichCompareBool(da.ptr(), db.ptr(), Py_EQ);
  if (r == -1)
    throw py::python_error();
  return r == 1;
}

template <typename T> py::object numpy_scalar_type_obj() {
  py::module_ np = py::module_::import_("numpy");
  if constexpr (std::is_same_v<T, int32_t>)
    return np.attr("int32");
  else if constexpr (std::is_same_v<T, uint32_t>)
    return np.attr("uint32");
  else if constexpr (std::is_same_v<T, int64_t>)
    return np.attr("int64");
  else if constexpr (std::is_same_v<T, uint64_t>)
    return np.attr("uint64");
  else if constexpr (std::is_same_v<T, float>)
    return np.attr("float32");
  else if constexpr (std::is_same_v<T, double>)
    return np.attr("float64");
  else if constexpr (std::is_same_v<T, triton_half>)
    return np.attr("float16");
}

template <typename T> bool numpy_dtype_matches(const py::object &dtype_obj) {
  return numpy_dtypes_equal(dtype_obj, numpy_scalar_type_obj<T>());
}

// This is a workaround because explicit template parameter list for lambdas is
// a C++20 extension:
// auto try_make_op = [&]<typename T>() {
//   if (dtype.is(pybind11::dtype::of<T>())) {
//     atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask,
//                                                      numel, order);
//   }
// };
template <RMWOp Op> struct OpCreator {
  py::object dtype;
  const uint64_t *ptr;
  const void *val;
  void *ret;
  const bool *mask;
  size_t numel;
  std::memory_order order;
  std::unique_ptr<AtomicOp> &atomic_op;

  template <typename T> void create() {
    if (!atomic_op && numpy_dtype_matches<T>(dtype)) {
      atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask,
                                                       numel, order);
    }
  }
};

template <RMWOp Op, typename... SupportedDTypes>
std::unique_ptr<AtomicOp>
makeAtomicRMWOp(py::object dtype, const uint64_t *ptr, const void *val,
                void *ret, const bool *mask, size_t numel,
                std::memory_order order) {
  // Iterate over all supported data types, make one that matches, and return
  std::unique_ptr<AtomicOp> atomic_op;
  OpCreator<Op> try_make_op{dtype, ptr,   val,   ret,
                            mask,  numel, order, atomic_op};

  (try_make_op.template create<SupportedDTypes>(), ...);
  if (!atomic_op) {
    throw std::invalid_argument("Unsupported data type");
  }
  // Make it a unique_ptr
  return atomic_op;
}

} // namespace

void init_triton_interpreter(py::module_ &m) {
  if (_import_array() < 0)
    throw py::python_error();

  py::enum_<MemSemantic>(m, "MEM_SEMANTIC")
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", MemSemantic::ACQUIRE)
      .value("RELEASE", MemSemantic::RELEASE)
      .value("RELAXED", MemSemantic::RELAXED)
      .export_values();

  py::enum_<RMWOp>(m, "RMW_OP")
      .value("ADD", RMWOp::ADD)
      .value("FADD", RMWOp::FADD)
      .value("AND", RMWOp::AND)
      .value("OR", RMWOp::OR)
      .value("XOR", RMWOp::XOR)
      .value("XCHG", RMWOp::XCHG)
      .value("MAX", RMWOp::MAX)
      .value("MIN", RMWOp::MIN)
      .value("UMIN", RMWOp::UMIN)
      .value("UMAX", RMWOp::UMAX)
      .export_values();

  m.def("load",
        [](py::object ptr_obj, py::object mask_obj, py::object other_obj,
           py::object ret_dtype_obj) -> py::object {
          py::module_ np = py::module_::import_("numpy");
          PyArrayObject *ptr_arr = require_ndarray(ptr_obj);
          require_ndarray(mask_obj);
          require_ndarray(other_obj);

          npy_intp numel = PyArray_SIZE(ptr_arr);
          int ndim = PyArray_NDIM(ptr_arr);
          npy_intp *dims = PyArray_DIMS(ptr_arr);

          py::object ret = np.attr("empty")(py::make_tuple(numel), ret_dtype_obj);
          PyArrayObject *ret_arr = require_ndarray(ret);

          py::object ptr_1d = np.attr("reshape")(ptr_obj, py::make_tuple(numel));
          py::object mask_1d = np.attr("reshape")(mask_obj, py::make_tuple(numel));
          py::object other_1d =
              np.attr("reshape")(other_obj, py::make_tuple(numel));

          PyArrayObject *rp = require_ndarray(ptr_1d);
          PyArrayObject *rm = require_ndarray(mask_1d);
          PyArrayObject *ro = require_ndarray(other_1d);

          auto *ptr_data = static_cast<uint64_t *>(PyArray_DATA(rp));
          auto *mask_data = static_cast<bool *>(PyArray_DATA(rm));
          void *other_data = PyArray_DATA(ro);
          void *ret_data = PyArray_DATA(ret_arr);
          size_t itemsize = static_cast<size_t>(PyArray_ITEMSIZE(ret_arr));

          for (npy_intp i = 0; i < numel; ++i) {
            void *dest = static_cast<char *>(ret_data) + i * itemsize;
            if (mask_data[i])
              memcpy(dest, reinterpret_cast<void *>(ptr_data[i]), itemsize);
            else
              memcpy(dest, static_cast<char *>(other_data) + i * itemsize,
                     itemsize);
          }

          py::list shape_list;
          for (int i = 0; i < ndim; ++i)
            shape_list.append(dims[i]);
          return np.attr("reshape")(ret, shape_list);
        });

  m.def("store",
        [](py::object ptr_obj, py::object value_obj, py::object mask_obj) {
          py::module_ np = py::module_::import_("numpy");
          PyArrayObject *ptr_arr = require_ndarray(ptr_obj);
          PyArrayObject *val_arr = require_ndarray(value_obj);
          require_ndarray(mask_obj);

          npy_intp numel = PyArray_SIZE(ptr_arr);

          py::object ptr_1d = np.attr("reshape")(ptr_obj, py::make_tuple(numel));
          py::object mask_1d = np.attr("reshape")(mask_obj, py::make_tuple(numel));
          py::object val_1d =
              np.attr("reshape")(value_obj, py::make_tuple(numel));

          PyArrayObject *rp = require_ndarray(ptr_1d);
          PyArrayObject *rm = require_ndarray(mask_1d);
          PyArrayObject *rv = require_ndarray(val_1d);

          auto *ptr_data = static_cast<uint64_t *>(PyArray_DATA(rp));
          auto *mask_data = static_cast<int8_t *>(PyArray_DATA(rm));
          void *val_data = PyArray_DATA(rv);
          size_t itemsize = static_cast<size_t>(PyArray_ITEMSIZE(val_arr));

          for (npy_intp i = 0; i < numel; ++i) {
            if (mask_data[i]) {
              memcpy(reinterpret_cast<void *>(ptr_data[i]),
                     static_cast<char *>(val_data) + i * itemsize, itemsize);
            }
          }
        });

  m.def("atomic_rmw",
        [](RMWOp rmw_op, py::object ptr_obj, py::object val_obj,
           py::object mask_obj, MemSemantic sem) -> py::object {
          py::module_ np = py::module_::import_("numpy");
          std::memory_order order = mem_semantic_map[sem];
          PyArrayObject *ptr_arr = require_ndarray(ptr_obj);
          require_ndarray(val_obj);
          require_ndarray(mask_obj);

          npy_intp numel = PyArray_SIZE(ptr_arr);
          int ndim = PyArray_NDIM(ptr_arr);
          npy_intp *dims = PyArray_DIMS(ptr_arr);

          py::object ret_dtype_obj = np.attr("dtype")(val_obj);
          py::object ret =
              np.attr("empty")(py::make_tuple(numel), ret_dtype_obj);
          PyArrayObject *ret_arr = require_ndarray(ret);

          py::object ptr_1d = np.attr("reshape")(ptr_obj, py::make_tuple(numel));
          py::object mask_1d = np.attr("reshape")(mask_obj, py::make_tuple(numel));
          py::object val_1d =
              np.attr("reshape")(val_obj, py::make_tuple(numel));

          PyArrayObject *reshaped_ptr = require_ndarray(ptr_1d);
          PyArrayObject *reshaped_mask = require_ndarray(mask_1d);
          PyArrayObject *reshaped_val = require_ndarray(val_1d);

          auto *ptr_data =
              static_cast<uint64_t *>(PyArray_DATA(reshaped_ptr));
          auto *mask_data = static_cast<bool *>(PyArray_DATA(reshaped_mask));
          auto *val_data = static_cast<const void *>(PyArray_DATA(reshaped_val));
          auto *ret_data = static_cast<void *>(PyArray_DATA(ret_arr));

          std::unique_ptr<AtomicOp> atomic_op;

#define MAKE_ATOMIC_RMW_OP(OP_NAME, ...)                                       \
  case OP_NAME:                                                                \
    atomic_op = makeAtomicRMWOp<OP_NAME, __VA_ARGS__>(                         \
        ret_dtype_obj, ptr_data, val_data, ret_data, mask_data, numel, order); \
    break;

          switch (rmw_op) {
            MAKE_ATOMIC_RMW_OP(RMWOp::ADD, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::FADD, triton_half, float, double)
            MAKE_ATOMIC_RMW_OP(RMWOp::AND, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::OR, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::XOR, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::MAX, int32_t, int64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::UMAX, uint32_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::MIN, int32_t, int64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::UMIN, uint32_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::XCHG, int32_t, uint32_t, int64_t,
                               uint64_t)
          default:
            throw std::invalid_argument("Unsupported RMW operation");
          }

#undef MAKE_ATOMIC_RMW_OP

          atomic_op->apply();

          py::list shape_list;
          for (int i = 0; i < ndim; ++i)
            shape_list.append(dims[i]);
          return np.attr("reshape")(ret, shape_list);
        });

  m.def("atomic_cas",
        [](py::object ptr_obj, py::object cmp_obj, py::object val_obj,
           MemSemantic sem) -> py::object {
          py::module_ np = py::module_::import_("numpy");
          std::memory_order order = mem_semantic_map[sem];
          PyArrayObject *ptr_arr = require_ndarray(ptr_obj);
          PyArrayObject *cmp_arr = require_ndarray(cmp_obj);
          require_ndarray(val_obj);

          npy_intp numel = PyArray_SIZE(ptr_arr);
          int ndim = PyArray_NDIM(ptr_arr);
          npy_intp *dims = PyArray_DIMS(ptr_arr);

          py::object ret_dtype_obj = np.attr("dtype")(cmp_obj);
          py::object ret =
              np.attr("empty")(py::make_tuple(numel), ret_dtype_obj);
          PyArrayObject *ret_arr = require_ndarray(ret);

          py::object ptr_1d = np.attr("reshape")(ptr_obj, py::make_tuple(numel));
          py::object cmp_1d = np.attr("reshape")(cmp_obj, py::make_tuple(numel));
          py::object val_1d =
              np.attr("reshape")(val_obj, py::make_tuple(numel));

          PyArrayObject *reshaped_ptr = require_ndarray(ptr_1d);
          PyArrayObject *reshaped_cmp = require_ndarray(cmp_1d);
          PyArrayObject *reshaped_val = require_ndarray(val_1d);

          size_t itemsize =
              static_cast<size_t>(PyArray_ITEMSIZE(reshaped_cmp));
          memcpy(PyArray_DATA(ret_arr), PyArray_DATA(reshaped_cmp),
                 itemsize * static_cast<size_t>(numel));
          AtomicCASOp(reinterpret_cast<const uint64_t *>(PyArray_DATA(reshaped_ptr)),
                      PyArray_DATA(ret_arr),
                      static_cast<const void *>(PyArray_DATA(reshaped_val)),
                      itemsize, static_cast<size_t>(numel), order)
              .apply();

          py::list shape_list;
          for (int i = 0; i < ndim; ++i)
            shape_list.append(dims[i]);
          return np.attr("reshape")(ret, shape_list);
        });
}
