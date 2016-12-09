#include <iostream>
using namespace std;

#include "matmul_cuda.h"

/* If stmt evaluates to false; error out */
#define ERR_RET(stmt) if (stmt) { cerr<<"Error - "<<__FILE__<<" "<<__LINE__<<endl; return; }

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


  /* Copy Results from GPU to CPU */
  err = cudaMemcpy(C, devC, data_bytes, cudaMemcpyDeviceToHost);
  ERR_RET(err != cudaSuccess);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return;
}
