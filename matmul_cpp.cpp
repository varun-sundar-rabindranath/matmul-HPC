#include <iostream>

#include <cstdlib> // malloc
#include <cassert> // assert

#include "timer.h"

using namespace std;

void matmul(int* A, int* B, int* C, int n) {

  for (int i = 0; i < n; i++) {     // Iterates the rows
    for (int j = 0; j < n; j++) {   // Iterates the columns

      int product = 0;
      for (int k = 0; k < n; k++) {
        product += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = product;

    }
  }

}

int main() {

  int mat_dim;

  cout<<"*********** Multiply square matrix **********"<<endl;
  cout<<"Enter matix dimension : ";
  cin>>mat_dim;
  cout<<endl;

  int* A = NULL; // input
  int* B = NULL; // input
  int* C = NULL; // A * B

  A = (int*) calloc(mat_dim * mat_dim, sizeof(int));
  B = (int*) calloc(mat_dim * mat_dim, sizeof(int));
  C = (int*) calloc(mat_dim * mat_dim, sizeof(int));

  assert(A != NULL && "Cannot allocate memory - A");
  assert(B != NULL && "Cannot allocate memory - B");

  /* Fill A and B */
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) { A[iter] = 1; B[iter] = 1; }

  startTimer();
  matmul(A, B, C, mat_dim); // C = A * B
  endTimer();

  /* Check C
   * Multiplying matrices of all 1's should leave C with all values 'mat_dim'
   */
  bool c_matmul_pass = true;
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) {
    if(C[iter] != mat_dim) { c_matmul_pass = false; break; }
  }

  /* Print result with elapsed time */
  if (c_matmul_pass) { cerr<<"C matmul pass. - "<<getElapsedTime()<<endl; }
                else { cerr<<"C matmul fail. - "<<getElapsedTime()<<endl; }

  free(A);
  free(B);
  free(C);

  return 0;
}