
#include "matmul_cuda.h"

#include <iostream>
using namespace std;

#include "utils.hpp" // transpose
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

/* matmul kernel transposed B */
__global__ void matmul_cuda_transpose(const float* A, const float* B, float* C, int mat_dim) {

  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

  /* Check bounds */
  if (idx_x < mat_dim && idx_y < mat_dim) {

    /* Calculate dot product */
    float product = 0;
    for (int dot_iter = 0; dot_iter < mat_dim; dot_iter++) {
      product += A[idx_y * mat_dim + dot_iter] * B[idx_x * mat_dim + dot_iter];
    }

    C[idx_y * mat_dim + idx_x] = product;
  }

}

/* matmul kernel transposed B */
__global__ void matmul_cuda_shared(const float* A, const float* B, float* C, int mat_dim) {

  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

  /* Get the stretched out 1D index */
  int blk_idx = threadIdx.y * blockDim.x + threadIdx.x;

  extern __shared__ float shared_bytes[];
  __shared__ float *smem_A;
  __shared__ float *smem_B;

  smem_A = shared_bytes;
  smem_B = &shared_bytes[blockDim.x * blockDim.y];

  /* Num of iterations for this tile size */
  int nbx = ((mat_dim - 1) / blockDim.x) + 1;

  int ax = threadIdx.x;
  int ay = idx_y;
  int bx = idx_x;
  int by = threadIdx.y;

  float product = 0;

  while (nbx--) {

    /* Load shared memroy */
    if (ax < mat_dim && ay < mat_dim) {
      smem_A[blk_idx] = A[ay * mat_dim + ax];
    } else {
      smem_A[blk_idx] = 0;
    }

    if (bx < mat_dim && by < mat_dim) {
      smem_B[blk_idx] = B[by * mat_dim + bx];
    } else {
      smem_B[blk_idx] = 0;
    }

    __syncthreads();

    for (int iter = 0; iter < blockDim.x; iter++) {
      product += smem_A[threadIdx.y * blockDim.x + iter] *
                 smem_B[iter * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    ax += blockDim.x;
    by += blockDim.y;
  }

  if (idx_x < mat_dim && idx_y < mat_dim) {
    C[idx_y * mat_dim + idx_x] = product;
  }
}

/* Host setup for GPU execution done here */
void run_matmul_cuda(const float* A, const float* B, float* C, int mat_dim, ULL* matmul_time) {

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
  dim3 grid_dim(((mat_dim - 1) / BLOCKDIM) + 1, ((mat_dim - 1) / BLOCKDIM) + 1);

  /* Invoke kernel and wait for its completion */
  startTimer(MATMUL_TIMER);
  matmul_cuda_naive<<<grid_dim, block_dim>>>(devA, devB, devC, mat_dim);
  cudaDeviceSynchronize();
  endTimer(MATMUL_TIMER);

  if (matmul_time) { *matmul_time = getElapsedTime(MATMUL_TIMER); }

#undef BLOCKDIM

  /* Copy Results from GPU to CPU */
  err = cudaMemcpy(C, devC, data_bytes, cudaMemcpyDeviceToHost);
  ERR_RET(err != cudaSuccess);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return;
}

/* Host setup for GPU execution done here */
void run_matmul_cuda_transpose(const float* A, const float* B, float* C, int mat_dim, ULL* matmul_time) {

  cudaError_t err;

  /* Transpose B matrix */
  float* B_transpose = NULL;
  B_transpose = (float*)calloc(mat_dim * mat_dim, sizeof(float));
  ERR_RET(B_transpose == NULL);

  transpose(B, B_transpose, mat_dim);

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
  err = cudaMemcpy(devB, B_transpose, data_bytes, cudaMemcpyHostToDevice);
  ERR_RET(err != cudaSuccess);

  /* Fix grid and block dim */
#define BLOCKDIM 32
  dim3 block_dim(BLOCKDIM, BLOCKDIM);
  dim3 grid_dim(((mat_dim - 1) / BLOCKDIM) + 1, ((mat_dim - 1) / BLOCKDIM) + 1);

  /* Invoke kernel and wait for its completion */
  startTimer(MATMUL_TIMER);
  matmul_cuda_transpose<<<grid_dim, block_dim>>>(devA, devB, devC, mat_dim);
  cudaDeviceSynchronize();
  endTimer(MATMUL_TIMER);

  if (matmul_time) { *matmul_time = getElapsedTime(MATMUL_TIMER); }

#undef BLOCKDIM

  /* Copy Results from GPU to CPU */
  err = cudaMemcpy(C, devC, data_bytes, cudaMemcpyDeviceToHost);
  ERR_RET(err != cudaSuccess);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return;
}

/* Host setup for GPU execution done here */
void run_matmul_cuda_shared(const float* A, const float* B, float* C, int mat_dim, ULL* matmul_time) {

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
  dim3 grid_dim(((mat_dim - 1) / BLOCKDIM) + 1, ((mat_dim - 1) / BLOCKDIM) + 1);
  size_t shared_bytes = block_dim.x * block_dim.y * 2 * sizeof(float);

  startTimer(MATMUL_TIMER);
  matmul_cuda_shared<<<grid_dim, block_dim, shared_bytes>>>(devA, devB, devC, mat_dim);
  cudaDeviceSynchronize();
  endTimer(MATMUL_TIMER);

  if (matmul_time) { *matmul_time = getElapsedTime(MATMUL_TIMER); }

#undef BLOCKDIM

  /* Copy Results from GPU to CPU */
  err = cudaMemcpy(C, devC, data_bytes, cudaMemcpyDeviceToHost);
  ERR_RET(err != cudaSuccess);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return;
}
