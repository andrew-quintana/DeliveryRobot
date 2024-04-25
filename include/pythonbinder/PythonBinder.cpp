#include <iostream>
#include <string>
#include <Python.h>

PythonBider::PythonBinder( std::string dir ) {
    // Initialize the Python interpreter
    Py_Initialize();

    // Set the path to the directory containing the Python script
    PyObject* sys_path = PySys_GetObject("path");
    PyList_Append(sys_path, Py_BuildValue("s", dir.c_str()));

}

int main() {
    
    Py_Initialize();

    // Set the path to the directory containing the Python script
    std::string python_script_dir = "path/to/directory/containing/python_script.py";
    PyObject* sys_path = PySys_GetObject("path");
    PyList_Append(sys_path, Py_BuildValue("s", python_script_dir.c_str()));

    // Import the Python module containing the function
    PyObject* module = PyImport_ImportModule("python_script");
    if (!module) {
        std::cerr << "Failed to import Python module" << std::endl;
        Py_Finalize();
        return 1;
    }

    // Get a reference to the Python function
    PyObject* py_function = PyObject_GetAttrString(module, "python_function");
    if (!py_function) {
        std::cerr << "Failed to get Python function" << std::endl;
        Py_DECREF(module);
        Py_Finalize();
        return 1;
    }

    // Call the Python function and get the result
    PyObject* args = Py_BuildValue("(i,f)", 5, 3.14);
    PyObject* result = PyObject_CallObject(py_function, args);
    if (!result) {
        std::cerr << "Failed to call Python function" << std::endl;
        Py_DECREF(py_function);
        Py_DECREF(module);
        Py_Finalize();
        return 1;
    }

    // Convert the result to a C++ float
    float float_result;
    PyArg_Parse(result, "f", &float_result);
    std::cout << "Result: " << float_result << std::endl;

    // Clean up
    Py_DECREF(result);
    Py_DECREF(py_function);
    Py_DECREF(module);
    Py_Finalize();

    return 0;
}
