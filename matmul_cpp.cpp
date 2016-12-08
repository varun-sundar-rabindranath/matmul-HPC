#include <iostream>

#include <cstdlib> // malloc
#include <cassert> // assert

#include "utils.hpp"
#include "timer.hpp"

using namespace std;

#define G 1000000000
#define M 1000000

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
  cout<<"Memory required : "<<(double)(mat_dim * mat_dim * 3 * sizeof(int)) /
                                                        (double)G<<" GB"<<endl;

  int* A = NULL; // input
  int* B = NULL; // input
  int* C = NULL; // A * B

  A = (int*) calloc(mat_dim * mat_dim, sizeof(int));
  B = (int*) calloc(mat_dim * mat_dim, sizeof(int));
  C = (int*) calloc(mat_dim * mat_dim, sizeof(int));

  assert(A != NULL && "Cannot allocate memory - A");
  assert(B != NULL && "Cannot allocate memory - B");
  assert(C != NULL && "Cannot allocate memory - C");

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
  if (c_matmul_pass) {
    cerr<<"C matmul pass. - "<<getElapsedTime() / (double)M<<" sec."<<endl;
  } else {
    cerr<<"C matmul fail. - "<<getElapsedTime() / (double)M<<" sec."<<endl;
  }

  free(A);
  free(B);
  free(C);

  return 0;
}
