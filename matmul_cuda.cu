
#include "matmul_cuda.h"

#include <iostream>
using namespace std;

#include "timer.hpp"

#define G 1000000000
#define M 1000000

/* If stmt evaluates to false; error out */
#define ERR_RET(stmt) if (stmt) { cerr<<"Error - "<<__FILE__<<" "<<__LINE__<<endl; return; }

/* matmul kernel naive */
__global__ void matmul_cuda_naive(const float* A, const float* B, float* C, int mat_dim) {

  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

  /* Check bounds */
  if (idx_x < mat_dim && idx_y < mat_dim) {

    /* Calculate dot product */
    float product = 0;
    for (int dot_iter = 0; dot_iter < mat_dim; dot_iter++) {
      product += A[idx_y * mat_dim + dot_iter] * B[dot_iter * mat_dim + idx_x];
    }

    C[idx_y * mat_dim + idx_x] = product;
  }

}

/* Host setup for GPU execution done here */
void run_matmul_cuda(const float* A, const float* B, float* C, int mat_dim) {

  cudaError_t err;

  float* devA = NULL;
  float* devB = NULL;
  float* devC = NULL;

  size_t data_bytes = mat_dim * mat_dim * sizeof(float);

  /* Allocate cuda device memory */
  err = cudaMalloc(&devA, data_bytes);
  ERR_RET(err != cudaSuccess)
  err = cudaMalloc(&devB, data_bytes);
  ERR_RET(err != cudaSuccess)
  err = cudaMalloc(&devC, data_bytes);
  ERR_RET(err != cudaSuccess)

  /* Copy Inputs from CPU to GPU */
  err = cudaMemcpy(devA, A, data_bytes, cudaMemcpyHostToDevice);
  ERR_RET(err != cudaSuccess);
  err = cudaMemcpy(devB, B, data_bytes, cudaMemcpyHostToDevice);
  ERR_RET(err != cudaSuccess);

  /* Fix grid and block dim */
#define BLOCKDIM 32
  dim3 block_dim(BLOCKDIM, BLOCKDIM);
  dim3 grid_dim((mat_dim / BLOCKDIM) + 1, (mat_dim / BLOCKDIM) + 1);

  startTimer();

  /* Invoke kernel and wait for its completion */
  matmul_cuda_naive<<<grid_dim, block_dim>>>(devA, devB, devC, mat_dim);
  cudaDeviceSynchronize();

  endTimer();
#undef BLOCKDIM

  /* Copy Results from GPU to CPU */
  err = cudaMemcpy(C, devC, data_bytes, cudaMemcpyDeviceToHost);
  ERR_RET(err != cudaSuccess);


  /* Multiplying matrices of all 1's should leave C with all values 'mat_dim' */
  bool matmul_pass = true;
  for (int iter = 0; iter < mat_dim * mat_dim; iter++) {
    if(C[iter] != mat_dim) { matmul_pass = false; break; }
  }

  /* Print result with elapsed time */
  if (matmul_pass) {
    cerr<<"CUDA matmul (Naive) pass - "<<
                                getElapsedTime() / (double)M<<" sec."<<endl;
  } else {
    cerr<<"CUDA matmul (Naive) fail - "<<
                                getElapsedTime() / (double)M<<" sec."<<endl;
  }

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return;
}
