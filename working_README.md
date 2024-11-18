# Accelerating a simple fluid sim with 2D CUDA FFTs
### UT ECE 361C (Multicore) Term Project
I've adapted the fluid sim so that it uses an abstract FFT type. This way we can slot in our own various FFT implementations (sequential, gpu, library).    
### Files to look at
See the abstract FFT type at `./fluid-solver-toy/src/FFT_Solver2d.h`, we will be extending this      
See `./src/Seq_FFT_Solver2d.h & .cpp` and `./src/GPU_FFT_Solver2d.h & .cpp` for the TODO implementations
See the last function in `./fluid-solver-toy/src/StamFFT_FluidSolver.cpp` to see how the fft is used and accessed           

### Notes on fft buffer format
      
FFTs will be performed on inputs of real #'s represented as float arrays        
the FFTs should be performed in place -- even if this is just a copy :(        
FFT buffer is of size (n*(n+2))        
        
[Here](https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html) is the MIT FFTW documentation on how their 2d ffts are formatted and stored -- let's format our FFTs the same way        
          
Where: for 0 <= j < N, 0 <= i < N in spatial domain and 0 <= i <= N/2 in fourier domain:         
real part of elem i,j: `buffer[i  +(N+2)*j]`    
imag part of elem i,j: `buffer[i+1+(N+2)*j]`       
       
These will be ffts of real data, thus there will be no imaginary part in the input       
