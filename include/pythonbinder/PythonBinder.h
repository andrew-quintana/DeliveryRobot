#ifndef PYTHONBINDER_H
#define PYTHONBINDER_H

#include <iostream>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

std::string command = "python3 -c \"import python_module; print(python_module.python_function(6, 3.14))\"";



#ifdef __cplusplus
}
#endif

#endif // PYTHONBINDER_H
