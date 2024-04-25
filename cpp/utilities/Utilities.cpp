//
// Created by Andrew Quintana on 1/6/24.
//

#include "Utilities.h"

#include <math.h>
#include <iostream>
#include <ctime>
#include <sstream>

using namespace std;

Component::Component() {
    this->logging = true;
    this->debug = true;
    this->verbose = false;
}
Component::~Component() {}

float deg_rad(float deg ) {
    return deg * (M_PI / 180);
}

void print_matrix( std::string matrix_name, const Eigen::MatrixXf& matrix ) {
    int rows = matrix.rows();
    int cols = matrix.cols();

    // Find the maximum width of each column
    std::vector<int> columnWidths(cols, 0);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            int width = std::to_string(matrix(row, col)).length();
            if (width > columnWidths[col]) {
                columnWidths[col] = width;
            }
        }
    }

    std::cout << matrix_name << std::endl;

    // Print the matrix with aligned columns
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << std::setw(columnWidths[col]) << matrix(row, col) << " ";
        }
        std::cout << std::endl;
    }
}

void insert_rows_and_cols(Eigen::MatrixXf& matrix, int n, int m, int r, int c) {

    // Create a new matrix with the desired size
    Eigen::MatrixXf newMatrix(matrix.rows() + r, matrix.cols() + c);
    newMatrix.setZero();

    // Copy the original matrix to the new matrix
    newMatrix.block(0, 0, n, m) = matrix.block(0, 0, n, m);
    newMatrix.block(n + r, 0, matrix.rows() - n, m) = matrix.block(n, 0, matrix.rows() - n, m);
    newMatrix.block(0, m + c, n, matrix.cols() - m) = matrix.block(0, m, n, matrix.cols() - m);
    newMatrix.block(n + r, m + c, matrix.rows() - n, matrix.cols() - m) = matrix.block(n, m, matrix.rows() - n, matrix.cols() - m);

    // Update the original matrix with the new matrix
    matrix = newMatrix;
}

void print_state(env& states ) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        std::cout << "ID:\t" << it->first << std::endl;
        std::cout << "\tX " << states[it->first](0);
        std::cout << "\tY " << states[it->first](1);
        std::cout << "\tPSI " << states[it->first](2) << std::endl;
    }
}

bool approximatelyEqual(float a, float b, float tolerance) {
    return std::abs(a - b) <= tolerance;
}

std::string format_string(const char* format, ...) {
    char buffer[256]; // Adjust the buffer size as needed
    va_list args;
    va_start(args, format);
    std::vsprintf(buffer, format, args);
    va_end(args);
    return std::string(buffer);
}

void status_update( int hyphen_sets, const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::string str = format_string(format, args);
    std::string hyphens = "- - - - - ";
    std::string prefix;
    for (int i = 0; i < hyphen_sets; ++i) {
        prefix += hyphens;
    }
    std::printf("\n%s - %s %s - - - - - - - - - - - - - - -\n",
                timestamp().c_str(),
                prefix.c_str(),
                str.c_str());
}

std::string timestamp() {
    std::time_t now = std::time(0);
    std::tm* timeinfo = std::localtime(&now);
    char timestamp[9];
    std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", timeinfo);

    // Print the timestamp
    return timestamp;

}