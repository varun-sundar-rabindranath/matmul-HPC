#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
using namespace std;

void print_matrix(float* mat, int dim, char* comment = NULL) {

  if (comment)
    cout<<comment<<endl;

  for (int y = 0; y < dim; y++) {
    for (int x = 0; x < dim; x++) {
      cout<<mat[y * dim + x]<<" ";
    }
    cout<<endl;
  }
}

void transpose(const float* A, float* At, int dim) {

  for (int h = 0; h < dim; h++) {
    for (int w = 0; w < dim; w++) {
      At[w * dim + h] = A[h * dim + w];
    }
  }

}

#endif // __UTILS_HPP__
