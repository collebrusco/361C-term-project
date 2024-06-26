#include "GPU_FFT_Solver.h"
#include <Stopwatch.h>
LOG_MODULE(gpufft);

#define PI 3.14159265358979323846
#define THREADS_PER_BLOCK 1024
#define TYPE_REAL float
#define TYPE_COMPLEX cuFloatComplex

static Stopwatch timer(MICROSECONDS);

__global__ static void fft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid  = threadIdx.x;

  int base = blockIdx.x * blockDim.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    revIndex |= (tempTid & 1);
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


__global__ static void invfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid  = threadIdx.x;

  int base = blockDim.x * blockIdx.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    revIndex |= (tempTid & 1);
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
        //   val = cuCdivf(cuCaddf(d_complex_in[(base+tid)], cuCmulf(twiddle, d_complex_in[(base+tid) + s])), make_cuFloatComplex(2.0, 0.0));
          val = cuCaddf(d_complex_in[(base+tid)], cuCmulf(twiddle, d_complex_in[(base+tid) + s]));
      }
      //odd
      else {
        //   val = cuCdivf(cuCsubf(d_complex_in[(base+tid) - s], cuCmulf(twiddle, d_complex_in[(base+tid)])), make_cuFloatComplex(2.0, 0.0));
          val = cuCsubf(d_complex_in[(base+tid) - s], cuCmulf(twiddle, d_complex_in[(base+tid)]));
      }
      __syncthreads();
      d_complex_in[(base+tid)] = val;
      __syncthreads();
  }
  d_in[2*(base+tid)] = cuCrealf(d_complex_in[(base+tid)]);
  d_in[2*(base+tid) + 1] = cuCimagf(d_complex_in[(base+tid)]);
}

__global__ static void cfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid = threadIdx.x;

  int base = blockIdx.x;
  int mul = blockDim.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    revIndex |= (tempTid & 1);
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


__global__ static void cinvfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
  int tid = threadIdx.x;

  int base = blockIdx.x;
  int mul = blockDim.x;

  int revIndex = 0;
  int tempTid = tid;
  for(int i = 1; i < size; i <<= 1) {
    revIndex <<= 1;
    revIndex |= (tempTid & 1);
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
        //   val = cuCdivf(cuCaddf(d_complex_in[(base+(mul*tid))], cuCmulf(twiddle, d_complex_in[base+(mul*(tid + s))])), make_cuFloatComplex(2.0, 0.0));
          val = cuCaddf(d_complex_in[(base+(mul*tid))], cuCmulf(twiddle, d_complex_in[base+(mul*(tid + s))]));
      }
      //odd
      else {
        //   val = cuCdivf(cuCsubf(d_complex_in[base+(mul*(tid - s))], cuCmulf(twiddle, d_complex_in[(base+(mul*tid))])), make_cuFloatComplex(2.0, 0.0));
          val = cuCsubf(d_complex_in[base+(mul*(tid - s))], cuCmulf(twiddle, d_complex_in[(base+(mul*tid))]));
      }
      __syncthreads();
      d_complex_in[(base+(mul*tid))] = val;
      __syncthreads();
  }
  d_in[2*(base+(mul*tid))] = cuCrealf(d_complex_in[(base+(mul*tid))]);
  d_in[2*(base+(mul*tid)) + 1] = cuCimagf(d_complex_in[(base+(mul*tid))]);
}


// ******** 1d ********
GPU_FFT_Solver1d::GPU_FFT_Solver1d(size_t n, float* buff) : FFT_Solver(n, buff) {
    din.malloc(N*2); dcin.malloc(N);
}

GPU_FFT_Solver1d::~GPU_FFT_Solver1d() {
}

float GPU_FFT_Solver1d::get_last_nocpy_us() {
    return lastt;
}
float GPU_FFT_Solver2d::get_last_nocpy_us() {
    return lastt;
}

void GPU_FFT_Solver1d::forward() {
    const unsigned int blocks = 1, threads = N;
    din.togpu(this->buffer);
    timer.reset_start();
    fft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    lastt = timer.stop(MICROSECONDS);
    din.tocpu(this->buffer);
}

void GPU_FFT_Solver1d::inverse() {
    const unsigned int blocks = 1, threads = N;
    din.togpu(this->buffer);
    timer.reset_start();
    invfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cinvfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    lastt = timer.stop(MICROSECONDS);
    din.tocpu(this->buffer);
}

// ******** 2d ********
GPU_FFT_Solver2d::GPU_FFT_Solver2d(size_t n, float* buff) : FFT_Solver(n, buff) {
    din.malloc(N*N*2); dcin.malloc(N*N);
}

GPU_FFT_Solver2d::~GPU_FFT_Solver2d() {
}

void GPU_FFT_Solver2d::forward() {
    const unsigned int blocks = N, threads = N;
    din.togpu(this->buffer);
    timer.reset_start();
    fft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    lastt = timer.stop(MICROSECONDS);
    din.tocpu(this->buffer);
}

void GPU_FFT_Solver2d::inverse() {
    const unsigned int blocks = N, threads = N;
    din.togpu(this->buffer);
    timer.reset_start();
    invfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cinvfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    lastt = timer.stop(MICROSECONDS);
    din.tocpu(this->buffer);
}

// #undef cu()
// static int en = 0;
// #define cu() \
// {cudaError_t err = cudaGetLastError();en++;\
// while (err != cudaSuccess) {\
//     printf("E %d: %s\n", en, cudaGetErrorString(err));\
//     err = cudaGetLastError(); \
// }}\

// #include <fstream>
// #include <iomanip>
// using namespace std;
// class filematrix {
// 	float* buf;
// public:
// 	float* const& buffer;
// 	const size_t n;
// 	filematrix(const char* fname, size_t size) : buffer(buf), n(size) {
// 		ifstream fin; 
// 		fin.open(fname);
// 		if (!fin) {LOG_ERR("oops (no file)");}
// 		buf = new float[n*n*2];
// 		for (int i = 0; i < n*n*2; i++)
// 			fin >> buf[i];
// 		fin.close();
// 	}
// 	~filematrix() {delete [] buf;}
// };

// #include <cmath>

// void outmat(string filename, float* matrix, size_t n) {
//     std::ofstream outFile(filename);
//     if (!outFile) {LOG_DBG("Error opening file: %s", filename); return;}

//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             // LOG_DBG("%.1f %.1f ", matrix[2 * (i * n + j)], matrix[2 * (i * n + j) + 1]);
//             outFile << matrix[2 * (i * n + j)] << " " << matrix[2 * (i * n + j) + 1] << " ";
//         }
//         outFile << std::endl;
//     }

//     outFile.close();
//     LOG_DBG("Matrix output to %s", filename.c_str());
// }

// void outbasic(const char* filename, size_t n) {
//     std::ofstream outFile(filename);
//     if (!outFile) {
//         LOG_DBG("Error opening file: %s", filename);
//         return;
//     }

//     int currentNumber = 0;

//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             outFile << currentNumber++ << " ";
//             outFile << currentNumber++ << " ";
//         }
//         outFile << std::endl;
//     }

//     outFile.close();
//     LOG_DBG("Matrix generated and saved to %s", filename);
// }

// float matdiff(float* matrix1, float* matrix2, size_t n) {
//     float sum = 0.0;

//     for (size_t i = 0; i < n * n; ++i) {
//         float realPartDiff = matrix1[2 * i] - matrix2[2 * i];
//         float imagPartDiff = matrix1[2 * i + 1] - matrix2[2 * i + 1];
//         float magnitude = sqrt(realPartDiff * realPartDiff + imagPartDiff * imagPartDiff);
//         sum += magnitude;
//     }

//     LOG_DBG("Sum of magnitudes of differences: %f", sum);
//     return sum;
// }

// #include <iostream>
// #include <fstream>
// #include <cmath>
// #include <iomanip>


// // Function to generate and output the matrix
// void sinmat(const char* filename, size_t n) {
//     std::ofstream outFile(filename);
//     if (!outFile) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }

//     // Setting precision for floating point numbers
//     outFile << std::fixed << std::setprecision(4);

//     for (size_t i = 0; i < n; ++i) { // For each row
//         double frequency = 2 * PI * (i + 1) / n; // Calculate frequency for i periods in a row
//         for (size_t j = 0; j < n; ++j) { // For each column
//             double x = static_cast<double>(j); // Normalized position in row
//             double sineValue = std::sin(frequency * x); // Calculate sine value
//             outFile << sineValue << " " << 0.0 << " "; // Output real part (sine value) and imaginary part (0)
//         }
//         outFile << std::endl;
//     }

//     outFile.close();
//     LOG_DBG("Matrix generated and saved to %s", filename);
// }


// __global__ static void placeidx(float* buffer, int n) {
//     int i = threadIdx.x + blockDim.x * blockIdx.x;
//     buffer[(2*i)] = (float)blockIdx.x; buffer[(2*i)+1] = (float)threadIdx.x;
// }

// #define obs(i, name) \
// {\
//     float *b = new float[i*i*2];\
//     \
//     cuda_buffer<float> ids; ids.malloc(i*i*2); ids.clear();\
//     placeidx<<<i,i>>>(ids.pt, i);\
//     memset(b,0,i*i*2*sizeof(float));\
//     ids.tocpu(b);\
//     outmat(#name, b, i); delete [] b;\
// }\

// static void forward(cuda_buffer<float>& din, cuda_buffer<cuFloatComplex>& dcin, size_t n) {
//     const unsigned int blocks = n;
//     const unsigned int threads = n;

//     fft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
//     cfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    

// }

// static void inverse(cuda_buffer<float>& din, cuda_buffer<cuFloatComplex>& dcin, size_t n) {
//     const unsigned int blocks = n;
//     const unsigned int threads = n;

//     invfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
//     cinvfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);

// }


// int ffttestmain() {

//     // obs(16, src/io/t16i.txt);
//     // obs(32, src/io/t32i.txt);
//     obs(1024, src/io/t1024i.txt);

//     sinmat("src/io/sm1024.txt", 1024);

//     cuda_buffer<float> din; din.malloc(1024*1024*2);
//     cuda_buffer<cuFloatComplex> dcin; dcin.malloc(1024*1024);

//     filematrix in("src/io/t1024i.txt", 1024);

//     din.togpu(in.buffer);

//     forward(din, dcin, 1024);
//     inverse(din, dcin, 1024);

//     float *b = new float[1024*1024*2];
//     din.tocpu(b);

//     outmat("src/io/f1024.txt", b, 1024);

//     filematrix c1("src/io/f1024.txt", 1024);
//     filematrix c2("src/io/t1024i.txt", 1024);

//     matdiff(c1.buffer, c2.buffer, 1024);

//     delete [] b;

// 	return 0;
// }

