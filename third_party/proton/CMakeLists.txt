cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0074 NEW)
project(Proton LANGUAGES CXX)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Check for includes =============
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
if(NOT TARGET Python::Module)
  add_library(Python::Module ALIAS Python3::Module)
endif()
set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
set(Python_SOABI "${Python3_SOABI}")
set(Python_VERSION "${Python3_VERSION}")
set(Python_INTERPRETER_ID "Python")
find_package(nanobind CONFIG REQUIRED)
nanobind_build_library(nanobind-static)
if(TRITON_STABLE_ABI)
  target_compile_definitions(nanobind-static PUBLIC Py_LIMITED_API=0x030C0000)
endif()
target_link_libraries(nanobind-static PRIVATE Python3::Module)

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module nanobind-static)

  # Use system to skip warnings caused by legacy clang compilers
  target_include_directories(${name}
    SYSTEM PRIVATE
      "${ROCTRACER_INCLUDE_DIR}"
  )

  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "PROTON_LIBS"
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_proton_library ============
add_subdirectory("${PROTON_COMMON_DIR}")
add_subdirectory("${PROTON_SRC_DIR}")

# ============ Add subdirectory with proton tests ============
add_subdirectory(test)

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_proton_obj_libs GLOBAL PROPERTY PROTON_LIBS)

if(NOT _proton_obj_libs)
  message(WARNING "No object libraries were defined in 'PROTON_LIBS'!")
endif()

set(_proton_obj_sources "")
foreach(_lib IN LISTS _proton_obj_libs)
  list(APPEND _proton_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${_proton_obj_sources})

target_link_libraries(proton PRIVATE Python3::Module nanobind-static)
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
