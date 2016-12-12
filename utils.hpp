#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
using namespace std;

void print_matrix(float* mat, int dim, char* comment = NULL);

void transpose(const float* A, float* At, int dim);

#endif // __UTILS_HPP__
