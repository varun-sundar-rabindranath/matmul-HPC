#include <iostream>

#include <cstdlib> // malloc
#include <cassert> // assert

#include <omp.h>   // openmp

#include "utils.hpp"
#include "timer.hpp"

#include "matmul_cuda.h" // Cuda matrix multiplication code

using namespace std;

#define G 1000000000
#define M 1000000

void matmul(const float* A, const float* B, float* C, int n) {

  for (int i = 0; i < n; i++) {     // Iterates the rows
    for (int j = 0; j < n; j++) {   // Iterates the columns

      float product = 0;
      for (int k = 0; k < n; k++) {
        product += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = product;

    }
  }

}

void matmul_transpose(const float* A, const float* B, float* C, int n) {

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {

      float product = 0;
      for (int k = 0; k < n; k++) {
        product += A[i * n + k] * B[j * n + k];
      }
      C[i * n + j] = product;

    }
  }

}

void matmul_transpose_multithread(const float* A, const float* B, float* C, int n) {

  omp_set_num_threads(6);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {

      float product = 0;
      for (int k = 0; k < n; k++) {
        product += A[i * n + k] * B[j * n + k];
      }
      C[i * n + j] = product;

    }
  }

}

void run_matmul_C(const float* A, const float* B, float* C, int mat_dim) {

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
  if (c_matmul_pass) {
    cerr<<"C matmul (Naive) pass - "<<getElapsedTime() / (double)M<<" sec."<<endl;
  } else {
    cerr<<"C matmul (Naive) fail - "<<getElapsedTime() / (double)M<<" sec."<<endl;
  }
}

void run_matmul_C_transpose(const float* A, const float* B, float* C, int mat_dim) {

  float* B_transpose = NULL;
  B_transpose = (float*)calloc(mat_dim * mat_dim, sizeof(float));
  assert(B_transpose != NULL && "Cannot allocate memory - B_transpose");

  transpose(B, B_transpose, mat_dim);

  startTimer();
  matmul_transpose(A, B_transpose, C, mat_dim); // C = A * B
  endTimer();

  /* Check C
   * Multiplying matrices of all 1's should leave C with all values 'mat_dim'
   */
  bool c_matmul_pass = true;
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) {
    if(C[iter] != mat_dim) { c_matmul_pass = false; break; }
  }

  /* Print result with elapsed time */
  if (c_matmul_pass) {
    cerr<<"C matmul (Transpose) pass - "<<
                                getElapsedTime() / (double)M<<" sec."<<endl;
  } else {
    cerr<<"C matmul (Transpose) fail - "<<
                                getElapsedTime() / (double)M<<" sec."<<endl;
  }

  free(B_transpose);
}

void run_matmul_C_transpose_multithread(const float* A, const float* B,
                                        float* C, int mat_dim) {

  float* B_transpose = NULL;
  B_transpose = (float*)calloc(mat_dim * mat_dim, sizeof(float));
  assert(B_transpose != NULL && "Cannot allocate memory - B_transpose");

  transpose(B, B_transpose, mat_dim);

  startTimer();
  matmul_transpose_multithread(A, B_transpose, C, mat_dim); // C = A * B
  endTimer();

  /* Check C
   * Multiplying matrices of all 1's should leave C with all values 'mat_dim'
   */
  bool c_matmul_pass = true;
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) {
    if(C[iter] != mat_dim) { c_matmul_pass = false; break; }
  }

  /* Print result with elapsed time */
  if (c_matmul_pass) {
    cerr<<"C matmul (Multithread transpose) pass - "<<
                                 getElapsedTime() / (double)M<<" sec."<<endl;
  } else {
    cerr<<"C matmul (Multithread transpose) fail - "<<
                                 getElapsedTime() / (double)M<<" sec."<<endl;
  }

  free(B_transpose);
}

int main(int argc, char *argv[]) {

  if(argc != 2) {
    /* Too many or too few arguments */
    cerr<<" Too many or too few arguments;"<<endl;
    cerr<<" Right usage is : ./matmul <square-matrix-dimension>"<<endl;
    return 0;
  }

  int mat_dim = atoi(argv[1]);

  cout<<"*********** Multiply square matrix **********"<<endl;
  cout<<"Dimension       : "<<mat_dim<<endl;
  cout<<"Memory required : "<<(double)(mat_dim * mat_dim * 3 * sizeof(float)) /
                                                        (double)G<<" GB"<<endl;

  float* A = NULL; // input
  float* B = NULL; // input
  float* C = NULL; // A * B

  A = (float*) calloc(mat_dim * mat_dim, sizeof(float));
  B = (float*) calloc(mat_dim * mat_dim, sizeof(float));
  C = (float*) calloc(mat_dim * mat_dim, sizeof(float));

  assert(A != NULL && "Cannot allocate memory - A");
  assert(B != NULL && "Cannot allocate memory - B");
  assert(C != NULL && "Cannot allocate memory - C");

  /* Fill A and B */
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) { A[iter] = 1; B[iter] = 1; }

  //run_matmul_C(A, B, C, mat_dim);

  //run_matmul_C_transpose(A, B, C, mat_dim);

  //run_matmul_C_transpose_multithread(A, B, C, mat_dim);

  run_matmul_cuda(A, B, C, mat_dim);

  free(A);
  free(B);
  free(C);

  return 0;
}
