#include <iostream>
#include <string>

int main() {
    // Execute the Python script
    std::string command = "python3 -c \"import python_module; print(python_module.python_function(6, 3.14))\"";
    int result = system(command.c_str());

    if (result == 0) {
        std::cout << "Python script executed successfully!" << std::endl;
    } else {
        std::cerr << "Error executing Python script." << std::endl;
    }

    return 0;
}