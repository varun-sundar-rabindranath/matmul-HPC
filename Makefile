matmul_cpp: matmul_cpp.cpp timer.cpp utils.hpp timer.hpp
	g++ -O3 matmul_cpp.cpp timer.cpp -o matmul_cpp -fopenmp
