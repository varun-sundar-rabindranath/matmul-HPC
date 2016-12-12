matmul: matmul_cpp.cpp matmul_cuda.h matmul_cuda.cu timer.cpp utils.hpp utils.cpp timer.hpp
	nvcc -ccbin g++ -c matmul_cuda.cu -o matmul_cuda.o
	g++ -O3 matmul_cpp.cpp timer.cpp utils.cpp matmul_cuda.o -o matmul -fopenmp -L/usr/local/cuda/lib64 -lcudart
