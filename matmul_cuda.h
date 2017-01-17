#ifndef __MATMUL_CUDA_H__
#define __MATMUL_CUDA_H__

void run_matmul_cuda(const float* A, const float* B, float* C, int mat_dim,
		     unsigned long long int* matmul_time = NULL);

void run_matmul_cuda_transpose(const float* A, const float* B, float* C, int mat_dim,
			       unsigned long long int* matmul_time = NULL);

void run_matmul_cuda_shared(const float* A, const float* B, float* C, int mat_dim,
			    unsigned long long int* matmul_time = NULL);

#endif // __MATMUL_CUDA_H__
