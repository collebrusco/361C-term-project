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
}

Seq_FFT_Solver1d::~Seq_FFT_Solver1d() {
}

void Seq_FFT_Solver1d::forward() {
    fft(buffer, N);
}

void Seq_FFT_Solver1d::inverse() {
    ifft(buffer, N);
}



Seq_FFT_Solver2d::Seq_FFT_Solver2d(size_t n, float* buff) : Seq_FFT_Solver1d(n, buff) {
    tbuff = new float[n*2];
}

Seq_FFT_Solver2d::~Seq_FFT_Solver2d() {
    delete [] tbuff;
}

void Seq_FFT_Solver2d::forward() {
    for (int row = 0; row < N; row++) {
        float* base = buffer + (row*N*2);
        memcpy(tbuff, base, N*2*sizeof(float));
        fft(tbuff, N);
        memcpy(base, tbuff, N*2*sizeof(float));
    }
    for (int col = 0; col < N; col++) {
        float* base = buffer + col*2;
        for (int elem = 0; elem < N; elem++) {
            tbuff[2*elem]       = base[((N*2) * elem)];
            tbuff[(2*elem)+1]   = base[((N*2) * elem)+1];
        }
        fft(tbuff, N);
        for (int elem = 0; elem < N; elem++) {
            base[((N*2) * elem)]    = tbuff[(2*elem)];
            base[((N*2) * elem)+1]  = tbuff[(2*elem)+1];
        }
    }
}

void Seq_FFT_Solver2d::inverse() {
    for (int row = 0; row < N; row++) {
        float* base = buffer + (row*N*2);
        memcpy(tbuff, base, N*2*sizeof(float));
        ifft(tbuff, N);
        memcpy(base, tbuff, N*2*sizeof(float));
    }
    for (int col = 0; col < N; col++) {
        float* base = buffer + col*2;
        for (int elem = 0; elem < N; elem++) {
            tbuff[2*elem]       = base[((N*2) * elem)];
            tbuff[(2*elem)+1]   = base[((N*2) * elem)+1];
        }
        ifft(tbuff, N);
        for (int elem = 0; elem < N; elem++) {
            base[((N*2) * elem)]    = tbuff[(2*elem)];
            base[((N*2) * elem)+1]  = tbuff[(2*elem)+1];
        }
    }
}

// #define SIZE (128)
// #define outarr(a) for (int i = 0; i < 2*SIZE; i++) cout << a[i] << " "; cout << "\n";
// #define adiff(a,b) {float sum = 0.; for (int i = 0; i < 2*SIZE*SIZE; i++) sum += (a[i]-b[i]) * (a[i]-b[i]); LOG_DBG("adiff %e", sum/(SIZE*SIZE));}
// #include <iostream>
// using namespace std;
// int main() {

//     float *arr = new float[2*SIZE*SIZE];
//     for (int i = 0; i < 2*SIZE*SIZE; i+=2) { arr[i] = i; arr[i + 1] = 0.; }
//     float *ref = new float[2*SIZE*SIZE];
//     for (int i = 0; i < 2*SIZE*SIZE; i+=2) { ref[i] = i; ref[i + 1] = 0.; }

//     Seq_FFT_Solver2d solver(SIZE, arr);
//     // outarr(arr);
//     solver.forward(); 
//     // outarr(arr);
//     solver.inverse();
//     for (int i = 0; i < 2*SIZE*SIZE; i+=2) { arr[i] /= SIZE*SIZE; }
//     // outarr(arr);
//     cout << "\n";

//     adiff(arr,ref);

//     delete [] arr;
//     delete [] ref;
//     return 0;
// }


// #define SIZE (128)
// #define outarr(a) for (int i = 0; i < 2*SIZE; i++) cout << a[i] << " "; cout << "\n";
// #define adiff(a,b) {float sum = 0.; for (int i = 0; i < 2*SIZE; i++) sum += (a[i]-b[i]) * (a[i]-b[i]); LOG_DBG("adiff %e", sum/SIZE);}
// #include <iostream>
// using namespace std;
// int main() {

//     float *arr = new float[2*SIZE];
//     for (int i = 0; i < 2*SIZE; i+=2) { arr[i] = i; arr[i + 1] = 0.; }
//     float *ref = new float[2*SIZE];
//     for (int i = 0; i < 2*SIZE; i+=2) { ref[i] = i; ref[i + 1] = 0.; }

//     Seq_FFT_Solver1d solver(SIZE, arr);
//     outarr(arr);
//     solver.forward(); 
//     outarr(arr);
//     solver.inverse();
//     for (int i = 0; i < 2*SIZE; i+=2) { arr[i] /= SIZE; }
//     outarr(arr);
//     cout << "\n";

//     adiff(arr,ref);

//     delete [] arr;
//     delete [] ref;
//     return 0;
// }
