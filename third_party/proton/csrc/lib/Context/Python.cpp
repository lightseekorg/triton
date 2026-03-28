#include "Context/Python.h"
#include "Utility/String.h"
#include <nanobind/nanobind.h>
#include <algorithm>
#include <string>

namespace proton {

namespace {

std::string unpackPyobject(PyObject *pyObject) {
  if (!pyObject)
    return "";
  if (PyBytes_Check(pyObject)) {
    Py_ssize_t size = 0;
    char *data = nullptr;
    PyBytes_AsStringAndSize(pyObject, &data, &size);
    if (data)
      return std::string(data, static_cast<size_t>(size));
    return "";
  }
  if (PyUnicode_Check(pyObject)) {
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize(pyObject, &size);
    if (!data)
      return "";
    return std::string(data, static_cast<size_t>(size));
  }
  return "";
}

std::string getAttrString(PyObject *obj, const char *attr) {
  PyObject *val = PyObject_GetAttrString(obj, attr);
  std::string result = unpackPyobject(val);
  Py_DecRef(val);
  return result;
}

long getAttrLong(PyObject *obj, const char *attr) {
  PyObject *val = PyObject_GetAttrString(obj, attr);
  long result = val ? PyLong_AsLong(val) : 0;
  Py_DecRef(val);
  return result;
}

} // namespace

std::vector<Context> PythonContextSource::getContextsImpl() {
  nanobind::gil_scoped_acquire gil;

  // sys._getframe() returns the current frame via stable API
  PyObject *sys = PyImport_ImportModule("sys");
  if (!sys)
    return {};
  PyObject *frame = PyObject_CallMethod(sys, "_getframe", nullptr);
  Py_DecRef(sys);
  if (!frame) {
    PyErr_Clear();
    return {};
  }

  std::vector<Context> contexts;
  while (frame && frame != Py_None) {
    PyObject *f_code = PyObject_GetAttrString(frame, "f_code");

    size_t lineno = static_cast<size_t>(getAttrLong(frame, "f_lineno"));
    std::string file = f_code ? getAttrString(f_code, "co_filename") : "";
    std::string function = f_code ? getAttrString(f_code, "co_name") : "";
    Py_DecRef(f_code);


    auto pythonFrame = formatFileLineFunction(file, lineno, function);
    contexts.push_back(Context(pythonFrame));

    PyObject *newFrame = PyObject_GetAttrString(frame, "f_back");
    Py_DecRef(frame);
    frame = newFrame;
  }
  Py_DecRef(frame);

  std::reverse(contexts.begin(), contexts.end());
  return contexts;
}

size_t PythonContextSource::getDepth() { return getContextsImpl().size(); }

} // namespace proton
