# Accelerating a Simple Fluid Sim with 2D CUDA FFTs
## Personal Project (Solver and Renderer) and UT ECE 361C Term Project (2D CUDA FFT)
![demo_gif](figs/product.gif)
     
Over 2024 Spring break, I was developing a 2D game that required a fluid simulation of the atmosphere. Researching simple fluid solvers, I came across [this 2001 paper by Jos Stam](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/jgt01.pdf). This solver is based around a 2D FFT which is used to low-pass the fluid vector field and force the field to be mass conserving, while a spatial domain advection technique gives the field its flow.

## The Solver
The solver algorithm itself is found in [StamFFT_FluidSolver.cpp](https://github.com/collebrusco/fluid-solver-toy/blob/absfft/src/StamFFT_FluidSolver.cpp), which you can reference to see un-interrupted code. Below I show the broad strokes of the solver function. First, the input forces are simply applied to the field, shown below.
```c++
    for ( i=0 ; i<N*N ; i++ ) {
        u[i] += dt*u0[i]; u0[i] = u[i];
        v[i] += dt*v0[i]; v0[i] = v[i];
    }
```
Next, the advection scheme is performed. The paper describes this scheme as "semi-Lagrangian".
```c++
    for ( i=0 ; i<N ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            x = i-dt*u0[i+N*j]*N; y = j-dt*v0[i+N*j]*N;
            i0 = __floor(x); s = x-i0; i0 = (N+(i0%N))%N; i1 = (i0+1)%N;
            j0 = __floor(y); t = y-j0; j0 = (N+(j0%N))%N; j1 = (j0+1)%N;
            u[i+N*j] = (1-s)*((1-t)*u0[i0+N*j0]+t*u0[i0+N*j1])+
                          s *((1-t)*u0[i1+N*j0]+t*u0[i1+N*j1]);
            v[i+N*j] = (1-s)*((1-t)*v0[i0+N*j0]+t*v0[i0+N*j1])+
                          s *((1-t)*v0[i1+N*j0]+t*v0[i1+N*j1]);
        }
    }
```
Next, in preperation for the spatial -> fourier domain transform, the 2D velocities are copied into the real part of the buffers on which we will perform the transform. `BUFF_R` and `BUFF_I` are macros to index the real and imaginary parts of the buffers.
```c++
    for ( i=0 ; i<N ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            BUFF_R(u0, i, j) = u[i+N*j]; 
            BUFF_R(v0, i, j) = v[i+N*j];
            BUFF_I(u0, i, j) = 0.; 
            BUFF_I(v0, i, j) = 0.;
        }
    }
```
Here is the spatial to fourier transform. This is the CUDA call that will be discussed later. It is abstracted so that a generic solver can be used (e.g. I use a library FFT to run the simulator on MacOS). After transforming to the fourier domain, a low-pass filter with a cutoff that is a function of viscosity is applied. At the same time, the fourier-domain vectors are projected to be tangent to concentric rings around the origin. The paper goes into more detail on how this removes the divergent component of the field. Afterwards, the fluid is transformed back into the spatial domain, and normalized. This is a complete step of the solver.
```c++
    fftu->forward(); fftv->forward();

    for (int i = 0; i < N; i++) {
        x = (i <= N/2) ? i : (float)i - (float)N;
        for (int j = 0; j < N; j++) {
            y = (j <= N/2) ? j : (float)j - (float)N;
            r = x*x + y*y;
            if (r == 0.0) continue;

            float *uf = &(u0[2*(i + N*j)]);
            float *vf = &(v0[2*(i + N*j)]);
            
            float f = exp(-r * dt * visc);
            
            float ur = f * ( (1 - x*x/r)*uf[0] - x*y/r * vf[0] );
            float ui = f * ( (1 - x*x/r)*uf[1] - x*y/r * vf[1] );
            float vr = f * ( -y*x/r * uf[0] + (1 - y*y/r)*vf[0] );
            float vi = f * ( -y*x/r * uf[1] + (1 - y*y/r)*vf[1] );

            (&(u0[2*(i + N*j)]))[0] = ur;
            (&(u0[2*(i + N*j)]))[1] = ui;
            (&(v0[2*(i + N*j)]))[0] = vr;
            (&(v0[2*(i + N*j)]))[1] = vi;
        }
    }
    fftu->inverse(); fftv->inverse();

    f = 1.0/(N*N);
    for ( i=0 ; i<N ; i++ ) {
        for ( j=0 ; j<N ; j++ )
        { 
            u[i+N*j] = f*BUFF_R(u0, i, j); v[i+N*j] = f*BUFF_R(v0, i, j); 
        }
    }

```
## The Renderer
The fluid simulator is not much good if you can't observe it! Moreover, I wanted to build an interactive demo of the solver that allows the user to swirl the fluid around. For this, I've used my [graphics library](https://github.com/collebrusco/flgl) that I've been maintaining for a few years. This library is simply OpenGL, GLFW, and some other conveniences.    
I built two renderers for the field. Both can be seen in the gif at the top of the page. The first simply places the x and y components of the vector into the red and blue color channels. This gives a surprisingly fluid like render. The renderer for this is ineffecient, but effective. I maintain a buffer of floating point x and y values, normalized to be between 0 and 1. This is buffered to the GPU every frame as a texture. The code for this (found [here](https://github.com/collebrusco/fluid-solver-toy/blob/absfft/src/rgFieldRenderer.cpp), using my flgl library) is below.
```c++
// upload sizeof(float)*2*n*n buffer to GPU
field_tex.bind();
field_tex.alloc(GL_RG32F, n, n, GL_RG, GL_FLOAT, field.buff());
field_tex.unbind();
// render texture to screen size quad
field_tex.bind();
field_tex.bind_to_unit(0);
field_shad.bind();
field_shad.uInt("uTexslot", 0);
gl.draw_mesh(quad);
field_tex.unbind();
field_shad.unbind();
```
The second renderer is the 'classic' vector field render style. The code can be found [here](https://github.com/collebrusco/fluid-solver-toy/blob/absfft/src/vecFieldRenderer.cpp). This is done by building a vertex buffer of lines originating from each vector pointing in their direction and scaled down by some constant. A line is only added every 4 vectors so that the final product is less messy. The CPU-side buffer is populated, sent to the GPU, and drawn every frame. This is much faster than the color renderer, though still requires high GPU bandwidth.
```c++

// build mesh
size_t bufi = 0;
for (size_t j = 0; j < (n); j+=4) {
    for (size_t i = 0; i < (n); i+=4) {

        // create vector vertices
        glm::vec2 vec = glm::vec2(u[i + j * n], v[i + j * n]);
        float l = glm::length(vec);
        vec *= (1./l);
        l = sqrt(l+1) - 1;
        l *= 2;
        l = max(l-0.1,0.);
        vec *= l;
        if (l<=0.) continue;

        // Scale positions to [-1, 1]
        buffer[bufi].pos = glm::vec2(-1.0f + size * i, -1.0f + size * j);

        // Calculate and scale vector
        buffer[bufi].vec = buffer[bufi].pos + coeff * vec; // Adjust vector by the given coefficient and add to position

        // Add line indices (this ultimately doesn't need this, should really be vbo only)
        size_t index = 2*bufi;
        indices.push_back(index); // start of line (position)
        indices.push_back(index+1); // end of line (position + vector)
        bufi++;
    }
}

// upload mesh to GPU
mesh.vao.bind();
mesh.vbo.bind();
mesh.vbo.buffer_data(bufi, buffer);
mesh.ibo.bind();
mesh.ibo.buffer(indices);
mesh.vao.unbind();
mesh.vbo.unbind();
mesh.ibo.unbind();
glLineWidth(2.0f);
glEnable(GL_LINE_SMOOTH);

// draw
line_shad.bind();
gl.draw_mesh(mesh, GL_LINES);
```

## The CUDA FFT Solver
Finally, we will look at the FFT solver itself, located [here](./src/GPU_FFT_Solver.cu). For simplicity, let's observe only the forward FFT. The FFT method is based on the original Cooley–Tukey method. Each thread block does a single row FFT, while each thread computes the result for one index of the row. The first step is reading in the input number for a given thread, which is done by bit-reversing the index. The data is then read into a GPU buffer of complex numbers, with proper synchronization.
```C++
__global__ static void invfft(TYPE_REAL *d_in, TYPE_COMPLEX *d_complex_in, int size) {
    int tid  = threadIdx.x;
    int base = blockDim.x * blockIdx.x;

    // bit reverse index
    int revIndex = 0;
    int tempTid = tid;
    for(int i = 1; i < size; i <<= 1) {
        revIndex <<= 1;
        revIndex |= (tempTid & 1);
        tempTid >>= 1;
    }

    // Load the input data using the reversed index and store it in a complex array
    TYPE_COMPLEX val = make_cuFloatComplex((TYPE_REAL)d_in[2*(base+revIndex)], (TYPE_REAL)d_in[2*(base+revIndex) + 1]);
    __syncthreads();
    d_complex_in[base+tid] = val;
    __syncthreads();
```
Next, the log(n) step is performed. This is the 'butterfly' step of the FFT. The loop is controlled by two variables: n (the number of elements being merged at each stage) and s (the step size that determines how far apart elements in the current stage are). Initially, n is set to 2, and s is set to 1, meaning that the first stage merges pairs of elements. In each iteration of the loop, n doubles (n <<= 1), and s also doubles (s <<= 1),  increasing the size of the groups being merged and halving the number of stages required to process the entire dataset. The loop continues until n exceeds the size of the dataset (size), at which point the entire dataset has been processed.

Within each iteration of the loop, the threads are grouped into smaller subgroups. Each thread computes its position within the current group using tid % n, which gives the index of the thread within the current subgroup of size n. The position of the thread relative to the half-size s determines whether it belongs to the even or odd part of the group. Depending on this, the thread either computes a sum (even part) or a difference (odd part) using the twiddle factor. The complex twiddle factor rotates the odd part of the data relative to the even part, allowing the FFT to efficiently combine these two parts at each stage. Finally, at each loop iteration, every thread calculates the new element, synchronizes, writes the element, then synchronizes again. This makes sure all the threads move together through the iterations.
```c++
    // n = # of elements being merged
    // s = step size
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
            val = cuCaddf(d_complex_in[(base+tid)], cuCmulf(twiddle, d_complex_in[(base+tid) + s]));
        }
        //odd
        else {
            val = cuCsubf(d_complex_in[(base+tid) - s], cuCmulf(twiddle, d_complex_in[(base+tid)]));
        }
        __syncthreads();
        d_complex_in[(base+tid)] = val;
        __syncthreads();
    }
    d_in[2*(base+tid)] = cuCrealf(d_complex_in[(base+tid)]);
    d_in[2*(base+tid) + 1] = cuCimagf(d_complex_in[(base+tid)]);
}

```
This kernel performs a number of 1D FFTs in parallel. To perform a 2D FFT, a 1D FFT is performed on each row, then each column. This kernel can be launched such that each thread block does a parallel row FFT. Next, a column FFT is performed with a different kernel. The only difference is the indexing scheme, such that the column kernel sees each thread block do a parallel column FFT.
```c++
// we need to track stride now
int tid = threadIdx.x;
int base = blockIdx.x;
int mul = blockDim.x;

// and this is how we index the 2D FFT buffer
__syncthreads();
d_complex_in[(base+(mul*tid))] = val;
__syncthreads();
```

Finally, the forward call looks like this.
```c++
void GPU_FFT_Solver2d::forward() {
    const unsigned int blocks = N, threads = N;
    din.togpu(this->buffer);

    fft<<<blocks, threads>>>(din.pt, dcin.pt, threads);
    cfft<<<blocks, threads>>>(din.pt, dcin.pt, threads);

    din.tocpu(this->buffer);
}
```
## Figures
![gpuvcpu_nocopy](./figs/FFTW%20v%20CUDA%20(no%20memcpy)%202D.png)     
Above is a comparison of FFTW's execution time vs our CUDA solver for the 2D case, not including buffering data.         

![gpuvcpu_copy](./figs/FFTW%20v%20CUDA%202D.png)     
Above is the same comparison including buffering. Even with this, the CUDA solver is a better bet for larger sizes. This is much less power effecient of course.
More figures can be found in `./figs`  

## Further work
There is no reason the entire solver shouldn't run on the GPU. This should be simple to implement, and would hugely reduce the gpu bandwidth overhead, which is huge, as the field has to go from the solver -> to GPU -> transform -> to CPU -> solver -> to GPU -> inverse -> to CPU -> finish solver. Even with this ineffeciency, it is faster than running the CPU solver (I used MIT's FFTW). Moving the solver to the GPU would hugely reduce GPU memcpys, at the expense of needing to read/write to VRAM in order to apply force to / read the velocity of the field. 

