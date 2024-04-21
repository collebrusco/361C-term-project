#include "GPU_FFT_Solver2d.h"
LOG_MODULE(gpufft);

#define PI 3.14159265358979323846
#define THREADS_PER_BLOCK 1024
#define TYPE_REAL float
#define TYPE_COMPLEX cuFloatComplex


__global__ void fft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid  = threadIdx.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    if(tempTid & 1) {
        revIndex |= 1;
    }
    tempTid >>= 1;
  }
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*revIndex], (TYPE_REAL)d_in[2*revIndex + 1]);
  __syncthreads();
	d_complex_in[tid] = val;
  __syncthreads();

  //n = # of elements being merged
  //s = step size
  for(int n = 2, s = 1; n <= size; n <<= 1, s <<= 1) {
      int idxInGroup = tid % n;
      //calculate twiddle
      int k = (idxInGroup < s) ? idxInGroup : idxInGroup - s;
      TYPE_REAL angle = -2.0 * PI * k / n;
      TYPE_COMPLEX twiddle = make_cuFloatComplex(cos(angle), sin(angle));
      TYPE_COMPLEX val;
      //split group into half
      //even
      if(idxInGroup < s) {
          val = cuCaddf(d_complex_in[tid], cuCmulf(twiddle, d_complex_in[tid + s]));
      }
      //odd
      else {
          val = cuCsubf(d_complex_in[tid - s], cuCmulf(twiddle, d_complex_in[tid]));
      }
      __syncthreads();
      d_complex_in[tid] = val;
      __syncthreads();
  }
  d_in[2*tid] = cuCrealf(d_complex_in[tid]);
  d_in[2*tid + 1] = cuCimagf(d_complex_in[tid]);
}


__global__ void invfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid  = threadIdx.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    if(tempTid & 1) {
        revIndex |= 1;
    }
    tempTid >>= 1;
  }
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*revIndex], (TYPE_REAL)d_in[2*revIndex + 1]);
  __syncthreads();
	d_complex_in[tid] = val;
  __syncthreads();

  //n = # of elements being merged
  //s = step size
  for(int n = 2, s = 1; n <= size; n <<= 1, s <<= 1) {
      int idxInGroup = tid % n;
      //calculate twiddle
      int k = (idxInGroup < s) ? idxInGroup : idxInGroup - s;
      TYPE_REAL angle = 2.0 * PI * k / n;
      TYPE_COMPLEX twiddle = make_cuFloatComplex(cos(angle), sin(angle));
      TYPE_COMPLEX val;
      //split group into half
      //even
      if(idxInGroup < s) {
          val = cuCdivf(cuCaddf(d_complex_in[tid], cuCmulf(twiddle, d_complex_in[tid + s])), make_cuFloatComplex(2.0, 0.0));
      }
      //odd
      else {
          val = cuCdivf(cuCsubf(d_complex_in[tid - s], cuCmulf(twiddle, d_complex_in[tid])), make_cuFloatComplex(2.0, 0.0));
      }
      __syncthreads();
      d_complex_in[tid] = val;
      __syncthreads();
  }
  d_in[2*tid] = cuCrealf(d_complex_in[tid]);
  d_in[2*tid + 1] = cuCimagf(d_complex_in[tid]);
}

#define ARRAY_BYTES (sizeof(float) * N * 2)

GPU_FFT_Solver2d::GPU_FFT_Solver2d(size_t n, float* buff) : FFT_Solver2d(n, buff) {
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_complex_in, N * sizeof(TYPE_COMPLEX));
	buffer_temp = (float*)malloc(N*N*2*sizeof(float));
}

GPU_FFT_Solver2d::~GPU_FFT_Solver2d() {
	cudaFree(d_in); cudaFree(d_complex_in);
	free(buffer_temp);
}

void GPU_FFT_Solver2d::forward() {
    const unsigned int threads = N;
    const unsigned int blocks = 1;

    //fft on every row
    for(int i = 0; i < N; i++) {
      // transfer the input array to the GPU
      cudaMemcpy(d_in, &buffer[2*i*N], ARRAY_BYTES, cudaMemcpyHostToDevice);

      //kernel call --> row fft
      fft<<<blocks, threads>>>(d_in, d_complex_in, threads);

      // copy back the result array to the CPU
      cudaMemcpy(&buffer_temp[2*i*N], d_in, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    }

    //fft on every col
    for(int i = 0; i < N; i++) {
      //tranpose
      for(int j = 0; j < N; j++) {
        cudaMemcpy(&d_in[2*j], &buffer_temp[2*(j*N + i)], sizeof(TYPE_REAL) * 2, cudaMemcpyHostToDevice);
      }

      //kernel call --> col fft
      fft<<<blocks, threads>>>(d_in, d_complex_in, threads);

      //transpose
      for(int j = 0; j < N; j++) {
        cudaMemcpy(&buffer[2*(j*N + i)], &d_in[2*j], sizeof(TYPE_REAL) * 2, cudaMemcpyDeviceToHost);
      }
    }
}

void GPU_FFT_Solver2d::inverse() {
    const unsigned int threads = N;
    const unsigned int blocks = 1;

    //inv fft on every row
    for(int i = 0; i < N; i++) {
      // transfer the input array to the GPU
      cudaMemcpy(d_in, &buffer[2*i*N], ARRAY_BYTES, cudaMemcpyHostToDevice);

      //kernel call --> inv fft
      invfft<<<blocks, threads>>>(d_in, d_complex_in, threads);

      // copy back the result array to the CPU
      cudaMemcpy(&buffer_temp[2*i*N], d_in, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    }

    //inv fft on every col
    for(int i = 0; i < N; i++) {
      //tranpose
      for(int j = 0; j < N; j++) {
        cudaMemcpy(&d_in[2*j], &buffer_temp[2*(j*N + i)], sizeof(TYPE_REAL) * 2, cudaMemcpyHostToDevice);
      }

      //kernel call --> col fft
      invfft<<<blocks, threads>>>(d_in, d_complex_in, threads);

      //transpose
      for(int j = 0; j < N; j++) {
        cudaMemcpy(&buffer[2*(j*N + i)], &d_in[2*j], sizeof(TYPE_REAL) * 2, cudaMemcpyDeviceToHost);
      }
    }

}

