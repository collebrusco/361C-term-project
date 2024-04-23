#include <flgl.h>
#include <flgl/tools.h>
#include <flgl/logger.h>
#include "SolverToy.h"
#include "FFT_Solver.h"
#include "GPU_FFT_Solver.h"
#include "Seq_FFT_Solver.h"
LOG_MODULE(testmain);
using namespace glm;
using namespace std;
#include <fstream>

#define N (512)

// static SolverToy toy(N);

// static unsigned int solver = 0;
// static GPU_FFT_Solver2d* gpu_fft;
// static Seq_FFT_Solver2d* seq_fft;
// static FFTW_FFT_Solver2d* fftw_fft;

// void upd(float dt) {
// 	(void)dt;
// 	if (window.keyboard[GLFW_KEY_L].pressed) {
// 		switch (solver&1) {
// 		case 0:
// 			LOG_DBG("SWITCHING SOLVER TO: FFTW");
// 			toy.set_fft_type(fftw_fft);
// 			break;
// 		case 1:
// 		// 	LOG_DBG("SWITCHING SOLVER TO: SEQUENTIAL");
// 		// 	toy.set_fft_type(seq_fft);
// 		// 	break;
// 		// case 2:
// 			LOG_DBG("SWITCHING SOLVER TO: GPU");
// 			toy.set_fft_type(gpu_fft);
// 			break;
// 		}
// 		solver++;
// 	}
// }

// int solmain() {

// 	glconfig.set_flgl_path("fluid-solver-toy/lib/flgl/");
// 	glconfig.set_shader_path("fluid-solver-toy/shaders/");

// 	toy.set_fft_type(gpu_fft);

// 	toy.run(upd);

// 	return 0;
// }



// int main() {


// 	return 0;
// }

