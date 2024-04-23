#ifndef GPU_FFT_SOLVER_H
#define GPU_FFT_SOLVER_H
#include <flgl.h>
#include <flgl/logger.h>
#include "FFT_Solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
class cuda_buffer {
    T* d_ptr;
    size_t _n;
public:
    T* const& pt;
    size_t const& n;
    cuda_buffer() : pt(d_ptr), n(_n) {d_ptr = 0;_n = 0;}
    ~cuda_buffer() {cudaFree(d_ptr);}
    void malloc(size_t size) {
        if (d_ptr != 0) cudaFree(d_ptr);
        _n = size;
        cudaMalloc((void**)&d_ptr, n * sizeof(T));
    }
    void clear() {cudaMemset(d_ptr, 0, n * sizeof(T));}
    void tocpu(T* buffer) {cudaMemcpy(buffer, d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost);}
    void togpu(T* buffer) {cudaMemcpy(d_ptr, buffer, n * sizeof(T), cudaMemcpyHostToDevice);}
};
/*
	the FFTs should be performed in place
	FFT buffer is of size (n*n*2)

	Where for i,j from 0 to N-1:
	real part of elem i,j: buffer[2 * (i+N*j)    ]
	imag part of elem i,j: buffer[2 * (i+N*j) + 1]
*/


/*
	The base class has members:
		const size_t N;
		float* const buffer;
*/

class GPU_FFT_Solver1d : public FFT_Solver {
private:
	cuda_buffer<float> din;
	cuda_buffer<cuFloatComplex> dcin;
public:
	GPU_FFT_Solver1d(size_t n, float* buff);
	virtual ~GPU_FFT_Solver1d();

	virtual void forward() override final;
	virtual void inverse() override final;
};

class GPU_FFT_Solver2d : public FFT_Solver {
private:
	cuda_buffer<float> din;
	cuda_buffer<cuFloatComplex> dcin;
public:
	GPU_FFT_Solver2d(size_t n, float* buff);
	virtual ~GPU_FFT_Solver2d();

	virtual void forward() override final;
	virtual void inverse() override final;
};

#endif
