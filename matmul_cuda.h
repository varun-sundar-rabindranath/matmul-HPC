#ifndef __MATMUL_CUDA_H__
#define __MATMUL_CUDA_H__

void run_matmul_cuda(const float* A, const float* B, float* C, int mat_dim);

void run_matmul_cuda_transpose(const float* A, const float* B, float* C, int mat_dim);

void run_matmul_cuda_shared(const float* A, const float* B, float* C, int mat_dim);

#endif // __MATMUL_CUDA_H__
