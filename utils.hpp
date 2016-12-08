#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
using namespace std;

void print_matrix(int* mat, int dim, char* comment = NULL) {

  if(comment)
    cout<<comment<<endl;

  for (int y = 0; y < dim; y++) {
    for (int x = 0; x < dim; x++) {
      cout<<mat[y * dim + x]<<" ";
    }
    cout<<endl;
  }
}

#endif // __UTILS_HPP__
