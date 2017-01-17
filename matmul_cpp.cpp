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

#define CHECK(A, B, C, mat_dim, type)                                           \
  cout<<"========== "<<type<<" =========="<<endl<<endl;                         \
  if(check_matmul(A, B, C, mat_dim)) {                                          \
    cout<<" PASS "<<getElapsedTime(APP_TIMER) / (double)M<<" sec."<<" | "       \
        <<"Matrix Multiply Time - "<<matmul_time / (double)M<<" sec. "<<endl;   \
  } else {                                                                      \
    cout<<" FAIL "<<getElapsedTime(APP_TIMER) / (double)M<<" sec."<<" | "       \
        <<"Matrix Multiply Time - "<<matmul_time / (double)M<<" sec. "<<endl;   \
  }                                                                             \
  cout<<endl;


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

void run_matmul_C(const float* A, const float* B, float* C, int mat_dim, ULL* matmul_time) {
  startTimer(MATMUL_TIMER);
  matmul(A, B, C, mat_dim); // C = A * B
  endTimer(MATMUL_TIMER);

  if (matmul_time) { *matmul_time = getElapsedTime(MATMUL_TIMER); }
}

void run_matmul_C_transpose(const float* A, const float* B, float* C, int mat_dim, ULL* matmul_time) {

  float* B_transpose = NULL;
  B_transpose = (float*)calloc(mat_dim * mat_dim, sizeof(float));
  assert(B_transpose != NULL && "Cannot allocate memory - B_transpose");

  transpose(B, B_transpose, mat_dim);

  startTimer(MATMUL_TIMER);
  matmul_transpose(A, B_transpose, C, mat_dim); // C = A * B
  endTimer(MATMUL_TIMER);

  if (matmul_time) { *matmul_time = getElapsedTime(MATMUL_TIMER); }

  free(B_transpose);
}

void run_matmul_C_transpose_multithread(const float* A, const float* B,
                                        float* C, int mat_dim, ULL* matmul_time) {

  float* B_transpose = NULL;
  B_transpose = (float*)calloc(mat_dim * mat_dim, sizeof(float));
  assert(B_transpose != NULL && "Cannot allocate memory - B_transpose");

  transpose(B, B_transpose, mat_dim);

  startTimer(MATMUL_TIMER);
  matmul_transpose_multithread(A, B_transpose, C, mat_dim); // C = A * B
  endTimer(MATMUL_TIMER);

  if (matmul_time) { *matmul_time = getElapsedTime(MATMUL_TIMER); }

  free(B_transpose);
}

bool check_matmul(const float* A, const float* B, float* C, int mat_dim) {

  for(int iter = 0; iter < mat_dim; iter++) {
    /* Test function for matrices A and B containing 1 */
    assert(A[iter] == 1 && B[iter] == 1);

    if(C[iter] != mat_dim) { return false; }
  }

  return true;
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
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) {
    A[iter] = 1; B[iter] = 1;
  }

  /* Get time taken for matrix multiply operation alone */
  ULL matmul_time;

  /********************************* Matrix multiply C *************************/
#if 1
  /* Naive matrix multiply */
  startTimer(APP_TIMER);
  run_matmul_C(A, B, C, mat_dim, &matmul_time);
  endTimer(APP_TIMER);

  CHECK(A, B, C, mat_dim, "C Naive Matrix Multiply")

  /* Transposed matrix multiply */
  startTimer(APP_TIMER);
  run_matmul_C_transpose(A, B, C, mat_dim, &matmul_time);
  endTimer(APP_TIMER);

  CHECK(A, B, C, mat_dim, "C Transpose Matrix Multiply")

  /* Transpose Multi-threaded Matrix Multiply */
  startTimer(APP_TIMER);
  run_matmul_C_transpose_multithread(A, B, C, mat_dim, &matmul_time);
  endTimer(APP_TIMER);

  CHECK(A, B, C, mat_dim, "C Transpose Multi-Threaded Matrix Multiply")
#endif

  /******************************** Matrix multiply CUDA ***********************/
  /* Naive CUDA Matrix Multiply */
#if 1
  startTimer(APP_TIMER);
  run_matmul_cuda(A, B, C, mat_dim, &matmul_time);
  endTimer(APP_TIMER);

  CHECK(A, B, C, mat_dim, "CUDA Naive Matrix Multiply")
#endif

  /* CUDA Transpose Matrix Multiply */

#if 1
  startTimer(APP_TIMER);
  run_matmul_cuda_transpose(A, B, C, mat_dim, &matmul_time);
  endTimer(APP_TIMER);

  CHECK(A, B, C, mat_dim, "CUDA Transpose Matrix Multiply");
#endif

#if 1
  startTimer(APP_TIMER);
  run_matmul_cuda_shared(A, B, C, mat_dim, &matmul_time);
  endTimer(APP_TIMER);

  CHECK(A, B, C, mat_dim, "CUDA Shared Memory Matrix Multiply");
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
