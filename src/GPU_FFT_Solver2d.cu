#include "GPU_FFT_Solver2d.h"
#include <Stopwatch.h>
LOG_MODULE(gpufft);

#define PI 3.14159265358979323846
#define THREADS_PER_BLOCK 1024
#define TYPE_REAL float
#define TYPE_COMPLEX cuFloatComplex

static Stopwatch timer(MICROSECONDS);

__global__ void offt(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
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
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*(revIndex)], (TYPE_REAL)d_in[2*(revIndex) + 1]);
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
  d_in[2*(tid)] = cuCrealf(d_complex_in[tid]);
  d_in[2*(tid) + 1] = cuCimagf(d_complex_in[tid]);
}

__global__ void fft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid  = threadIdx.x;

  int base = blockIdx.x * blockDim.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    if(tempTid & 1) {
        revIndex |= 1;
    }
    tempTid >>= 1;
  }
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*(base+revIndex)], (TYPE_REAL)d_in[2*(base+revIndex) + 1]);
  __syncthreads();
	d_complex_in[base+tid] = val;
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
          val = cuCaddf(d_complex_in[base+tid], cuCmulf(twiddle, d_complex_in[base+tid + s]));
      }
      //odd
      else {
          val = cuCsubf(d_complex_in[base+tid - s], cuCmulf(twiddle, d_complex_in[base+tid]));
      }
      __syncthreads();
      d_complex_in[base+tid] = val;
      __syncthreads();
  }
  d_in[2*(base+tid)] = cuCrealf(d_complex_in[base+tid]);
  d_in[2*(base+tid) + 1] = cuCimagf(d_complex_in[base+tid]);
}


__global__ void invfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid  = threadIdx.x;

  int base = blockDim.x * blockIdx.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    if(tempTid & 1) {
        revIndex |= 1;
    }
    tempTid >>= 1;
  }
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*(base+revIndex)], (TYPE_REAL)d_in[2*(base+revIndex) + 1]);
  __syncthreads();
	d_complex_in[base+tid] = val;
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
          val = cuCdivf(cuCaddf(d_complex_in[(base+tid)], cuCmulf(twiddle, d_complex_in[(base+tid) + s])), make_cuFloatComplex(2.0, 0.0));
      }
      //odd
      else {
          val = cuCdivf(cuCsubf(d_complex_in[(base+tid) - s], cuCmulf(twiddle, d_complex_in[(base+tid)])), make_cuFloatComplex(2.0, 0.0));
      }
      __syncthreads();
      d_complex_in[(base+tid)] = val;
      __syncthreads();
  }
  d_in[2*(base+tid)] = cuCrealf(d_complex_in[(base+tid)]);
  d_in[2*(base+tid) + 1] = cuCimagf(d_complex_in[(base+tid)]);
}

__global__ void cfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid = threadIdx.x;

  int base = blockIdx.x;
  int mul = blockDim.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    if(tempTid & 1) {
        revIndex |= 1;
    }
    tempTid >>= 1;
  }
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*(base+(mul*revIndex))], (TYPE_REAL)d_in[2*(base+(mul*revIndex)) + 1]);
  __syncthreads();
	d_complex_in[base+(tid*mul)] = val;
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
          val = cuCaddf(d_complex_in[base+(mul*tid)], cuCmulf(twiddle, d_complex_in[base+(mul*(tid + s))]));
      }
      //odd
      else {
          val = cuCsubf(d_complex_in[base+(mul*(tid - s))], cuCmulf(twiddle, d_complex_in[base+(mul*tid)]));
      }
      __syncthreads();
      d_complex_in[base+(mul*tid)] = val;
      __syncthreads();
  }
  d_in[2*(base+(mul*tid))] = cuCrealf(d_complex_in[base+(mul*tid)]);
  d_in[2*(base+(mul*tid)) + 1] = cuCimagf(d_complex_in[base+(mul*tid)]);
}


__global__ void cinvfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid = threadIdx.x;

  int base = blockIdx.x;
  int mul = blockDim.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    if(tempTid & 1) {
        revIndex |= 1;
    }
    tempTid >>= 1;
  }
  TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*(base+(mul*revIndex))], (TYPE_REAL)d_in[2*(base+(mul*revIndex)) + 1]);
  __syncthreads();
	d_complex_in[base+(mul*tid)] = val;
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
          val = cuCdivf(cuCaddf(d_complex_in[(base+(mul*tid))], cuCmulf(twiddle, d_complex_in[base+(mul*(tid + s))])), make_cuFloatComplex(2.0, 0.0));
      }
      //odd
      else {
          val = cuCdivf(cuCsubf(d_complex_in[base+(mul*(tid - s))], cuCmulf(twiddle, d_complex_in[(base+(mul*tid))])), make_cuFloatComplex(2.0, 0.0));
      }
      __syncthreads();
      d_complex_in[(base+(mul*tid))] = val;
      __syncthreads();
  }
  d_in[2*(base+(mul*tid))] = cuCrealf(d_complex_in[(base+(mul*tid))]);
  d_in[2*(base+(mul*tid)) + 1] = cuCimagf(d_complex_in[(base+(mul*tid))]);
}

__global__ void oinvfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
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

static int en = 0;
#define cu() \
{cudaError_t err = cudaGetLastError();en++;\
while (err != cudaSuccess) {\
    printf("E %d: %s\n", en, cudaGetErrorString(err));\
    err = cudaGetLastError(); \
}}\

#include <fstream>
#include <iomanip>
using namespace std;
class filematrix {
	float* buf;
public:
	float* const& buffer;
	const size_t n;
	filematrix(const char* fname, size_t size) : buffer(buf), n(size) {
		ifstream fin; 
		fin.open(fname);
		if (!fin) {LOG_ERR("oops (no file)");}
		buf = new float[n*n*2];
		for (int i = 0; i < n*n*2; i++)
			fin >> buf[i];
		fin.close();
	}
	~filematrix() {delete [] buf;}
};

#include <cmath>

void outmat(string filename, float* matrix, size_t n) {
    std::ofstream outFile(filename);
    if (!outFile) {LOG_DBG("Error opening file: %s", filename); return;}

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // LOG_DBG("%.1f %.1f ", matrix[2 * (i * n + j)], matrix[2 * (i * n + j) + 1]);
            outFile << matrix[2 * (i * n + j)] << " " << matrix[2 * (i * n + j) + 1] << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
    LOG_DBG("Matrix output to %s", filename);
}

void outbasic(const char* filename, size_t n) {
    std::ofstream outFile(filename);
    if (!outFile) {
        LOG_DBG("Error opening file: %s", filename);
        return;
    }

    int currentNumber = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            outFile << currentNumber++ << " ";
            outFile << currentNumber++ << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
    LOG_DBG("Matrix generated and saved to %s", filename);
}

float matdiff(float* matrix1, float* matrix2, size_t n) {
    float sum = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        float realPartDiff = matrix1[2 * i] - matrix2[2 * i];
        float imagPartDiff = matrix1[2 * i + 1] - matrix2[2 * i + 1];
        float magnitude = sqrt(realPartDiff * realPartDiff + imagPartDiff * imagPartDiff);
        sum += magnitude;
    }

    LOG_DBG("Sum of magnitudes of differences: %f", sum);
    return sum;
}

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>


// Function to generate and output the matrix
void sinmat(const char* filename, size_t n) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Setting precision for floating point numbers
    outFile << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < n; ++i) { // For each row
        double frequency = 2 * PI * (i + 1) / n; // Calculate frequency for i periods in a row
        for (size_t j = 0; j < n; ++j) { // For each column
            double x = static_cast<double>(j); // Normalized position in row
            double sineValue = std::sin(frequency * x); // Calculate sine value
            outFile << sineValue << " " << 0.0 << " "; // Output real part (sine value) and imaginary part (0)
        }
        outFile << std::endl;
    }

    outFile.close();
    LOG_DBG("Matrix generated and saved to %s", filename);
}


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
        cudaMalloc((void**)&d_ptr, n * sizeof(T)); cu();
    }
    void clear() {cudaMemset(d_ptr, 0, n * sizeof(T)); cu();}
    void tocpu(T* buffer) {cudaMemcpy(buffer, d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost); cu();}
    void togpu(T* buffer) {cudaMemcpy(d_ptr, buffer, n * sizeof(T), cudaMemcpyHostToDevice); cu();}
};

__global__ void placeidx(float* buffer, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    buffer[(2*i)] = (float)blockIdx.x; buffer[(2*i)+1] = (float)threadIdx.x;
}

#define obs(i, name) \
{\
    float b[i*i*2];\
    \
    cuda_buffer<float> ids; ids.malloc(i*i*2); ids.clear();\
    placeidx<<<i,i>>>(ids.pt, i);\
    memset(b,0,i*i*2*sizeof(float));\
    ids.tocpu(b);\
    outmat(#name, b, i);\
}\

#define blocks 256
static void forward(cuda_buffer<float>& din, cuda_buffer<cuFloatComplex>& dcin, size_t n) {
    const unsigned int threads = n;

    fft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    

}

static void inverse(cuda_buffer<float>& din, cuda_buffer<cuFloatComplex>& dcin, size_t n) {
    const unsigned int threads = n;

    invfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cinvfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);

}


int main() {

    // obs(16, src/io/t16i.txt);
    // obs(32, src/io/t32i.txt);
    obs(256, src/io/t256i.txt);

    sinmat("src/io/sm256.txt", 256);

    cuda_buffer<float> din; din.malloc(256*256*2);
    cuda_buffer<cuFloatComplex> dcin; dcin.malloc(256*256);

    filematrix in("src/io/t256i.txt", 256);

    din.togpu(in.buffer);

    forward(din, dcin, 256);
    inverse(din, dcin, 256);

    float *b = new float[256*256*2];
    din.tocpu(b);

    outmat("src/io/f256.txt", b, 256);

    filematrix c1("src/io/f256.txt", 256);
    filematrix c2("src/io/t256i.txt", 256);

    matdiff(c1.buffer, c2.buffer, 256);

    delete [] b;

	return 0;
}


