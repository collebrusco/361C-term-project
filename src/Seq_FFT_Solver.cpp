#include "Seq_FFT_Solver.h"
#include <complex>
LOG_MODULE(seqfft);

#define PI (3.141592653589793238462643383279502884197)


void Seq_FFT_Solver1d::fft(float* x, const size_t n) {
    // Check if it is splitted enough
    if (n <= 1) {
        return;
    }

    // Split even and odd
    float *odd = new float[2*(n/2)];
    float *even = new float[2*(n/2)];
    for (int i = 0; i < n / 2; i++) {
        even[(2*i)]     = x[2*(i*2)    ];
        even[(2*i)+1]   = x[2*(i*2)  +1];
        odd[(2*i)]      = x[2*(i*2+1)  ];
        odd[(2*i)+1]    = x[2*(i*2+1)+1];
    }

    // Split on tasks
    fft(even, n/2);
    fft(odd, n/2);

    // Calculate DFT
    for (int k = 0; k < n / 2; k++) {
        std::complex<float> t = exp(std::complex<float>(0, -2 * PI * k / n)) * std::complex<float>(odd[(2*k)], odd[(2*k)+1]);
        x[(2*k)] =  even[(2*k)] + t.real();
        x[(2*k)+1] = even[(2*k)+1] + t.imag();
        x[2*(n / 2 + k)] = even[(2*k)] - t.real();
        x[2*(n / 2 + k)+1] = even[(2*k)+1] - t.imag();
    }
    delete [] even; delete [] odd;
}

void Seq_FFT_Solver1d::ifft(float* x, const size_t n) {
    // Check if it is splitted enough
    if (n <= 1) {
        return;
    }

    // Split even and odd
    float *odd = new float[2*(n/2)];
    float *even = new float[2*(n/2)];
    for (int i = 0; i < n / 2; i++) {
        even[(2*i)]     = x[2*(i*2)    ];
        even[(2*i)+1]   = x[2*(i*2)  +1];
        odd[(2*i)]      = x[2*(i*2+1)  ];
        odd[(2*i)+1]    = x[2*(i*2+1)+1];
    }

    // Split on tasks
    ifft(even, n/2);
    ifft(odd, n/2);

    
    // Calculate DFT
    for (int k = 0; k < n / 2; k++) {
        std::complex<float> t = exp(std::complex<float>(0, 2 * PI * k / n)) * std::complex<float>(odd[(2*k)], odd[(2*k)+1]);
        x[(2*k)] =  even[(2*k)] + t.real();
        x[(2*k)+1] = even[(2*k)+1] + t.imag();
        x[2*(n / 2 + k)] = even[(2*k)] - t.real();
        x[2*(n / 2 + k)+1] = even[(2*k)+1] - t.imag();
    }
    delete [] even; delete [] odd;
}

Seq_FFT_Solver1d::Seq_FFT_Solver1d(size_t n, float* buff) : FFT_Solver(n, buff) {
	// TODO constructor
}

Seq_FFT_Solver1d::~Seq_FFT_Solver1d() {
	// TODO destructor
}

void Seq_FFT_Solver1d::forward() {
    fft(buffer, N);
}

void Seq_FFT_Solver1d::inverse() {
    ifft(buffer, N);
}



Seq_FFT_Solver2d::Seq_FFT_Solver2d(size_t n, float* buff) : FFT_Solver(n, buff) {
	// TODO constructor
}

Seq_FFT_Solver2d::~Seq_FFT_Solver2d() {
	// TODO destructor
}

void Seq_FFT_Solver2d::forward() {
	// TODO fwd
}

void Seq_FFT_Solver2d::inverse() {
	// TODO inv
}

#define outarr() for (int i = 0; i < 16; i++) cout << arr[i] << " "; cout << "\n";
#include <iostream>
using namespace std;
int main() {

    float *arr = new float[16];
    for (int i = 0; i < 16; i+=2) { arr[i] = i; arr[i + 1] = 0.; }

    Seq_FFT_Solver1d solver(8, arr);
    outarr();
    solver.forward(); 
    outarr();
    solver.inverse();
    for (int i = 0; i < 16; i+=2) { arr[i] /= 8.; }
    outarr();
    cout << "\n";

    delete [] arr;
    return 0;
}
