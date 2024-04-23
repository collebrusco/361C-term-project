#include <flgl.h>
#include <flgl/tools.h>
#include <flgl/logger.h>
#include "SolverToy.h"
#include "FFT_Solver.h"
#include "GPU_FFT_Solver.h"
#include "Seq_FFT_Solver.h"
#include <Stopwatch.h>
LOG_MODULE(testmain);
using namespace glm;
using namespace std;
#include <fstream>

// #define N (512)

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
// 			LOG_DBG("SWITCHING SOLVER TO: SEQUENTIAL");
// 			toy.set_fft_type(seq_fft);
// 			break;
// 		case 1:
// 			LOG_DBG("SWITCHING SOLVER TO: FFTW");
// 			toy.set_fft_type(fftw_fft);
// 			break;
// 		case 2:
// 			LOG_DBG("SWITCHING SOLVER TO: GPU");
// 			toy.set_fft_type(gpu_fft);
// 			break;
// 		}
// 		solver++;
// 	}
// }

// int main() {

// 	glconfig.set_flgl_path("fluid-solver-toy/lib/flgl/");
// 	glconfig.set_shader_path("fluid-solver-toy/shaders/");

// 	toy.set_fft_type(seq_fft);

// 	toy.run(upd);

// 	return 0;
// }

static unsigned int M = 0xF28D8B38;
static float random() {return (float)(0x123FF941*M+0xABABEE77);}
static Stopwatch timer(MICROSECONDS);

static void test(FFT_Solver* solver, string name, ofstream& fout) {
	timer.stop_reset();
	float buf[64];
	for (int i = 0; i < 64; i++) {
		timer.reset_start();
		solver->forward(); solver->inverse();
		buf[i] = timer.stop();
	}
	float mean = 0., sig = 0.;
	for (int i = 0; i < 64; i++)
		mean += buf[i];
	mean /= 64.;
	for (int i = 0; i < 64; i++)
		sig += (buf[i]-mean) * (buf[i]-mean);
	sig /= 64.;
	sig = sqrt(sig);
	fout << name << " m " << mean << " s " << sig << "\n";
}

int main() {

	FFTW_FFT_Solver1d *fftw1d;
	FFTW_FFT_Solver2d *fftw2d;

	Seq_FFT_Solver1d * seq1d;
	Seq_FFT_Solver2d * seq2d;

	GPU_FFT_Solver1d * gpu1d;
	GPU_FFT_Solver2d * gpu2d;

	ofstream fout; fout.open("src/io/RESULTS.txt"); if (!fout) return -666;

	for (unsigned int n = 8; n <= 1024; n <<= 1) {
		LOG_DBG("======== n = %d ========", n);
		float* buff = new float[n*n*2];
		for (int i = 0; i < n*n*2; i++) buff[i] = random();
		fftw1d = new FFTW_FFT_Solver1d(n, buff);
		fftw2d = new FFTW_FFT_Solver2d(n, buff);
		seq1d  = new Seq_FFT_Solver1d(n, buff);
		seq2d  = new Seq_FFT_Solver2d(n, buff);
		gpu1d  = new GPU_FFT_Solver1d(n, buff);
		gpu2d  = new GPU_FFT_Solver2d(n, buff);

		fout << n << "\n";

		test(fftw1d, "fftw1d", fout);
		for (int i = 0; i < n*n*2; i++) buff[i] = random();
		test(fftw2d, "fftw2d", fout);
		for (int i = 0; i < n*n*2; i++) buff[i] = random();
		test(seq1d, "seq1d", fout);
		for (int i = 0; i < n*n*2; i++) buff[i] = random();
		test(seq2d, "seq2d", fout);
		for (int i = 0; i < n*n*2; i++) buff[i] = random();
		test(gpu1d, "gpu1d", fout);
		for (int i = 0; i < n*n*2; i++) buff[i] = random();
		test(gpu2d, "gpu2d", fout);
		for (int i = 0; i < n*n*2; i++) buff[i] = random();

		fout << "\n";

		// timer.reset_start();
		// fftw1d->forward(); fftw1d->inverse();
		// LOG_DBG("fftw1d: %.3fus", timer.stop());
		// timer.reset_start();
		// fftw2d->forward(); fftw2d->inverse();
		// LOG_DBG("fftw2d: %.3fus", timer.stop());
		
		// timer.reset_start();
		// seq1d->forward(); seq1d->inverse();
		// LOG_DBG("seq1d: %.3fus", timer.stop());
		// timer.reset_start();
		// seq2d->forward(); seq2d->inverse();
		// LOG_DBG("seq2d: %.3fus", timer.stop());
		
		// timer.reset_start();
		// gpu1d->forward(); gpu1d->inverse();
		// LOG_DBG("gpu1d: %.3fus", timer.stop());
		// timer.reset_start();
		// gpu2d->forward(); gpu2d->inverse();
		// LOG_DBG("gpu2d: %.3fus", timer.stop());

		LOG_DBG("\n");

		delete fftw1d; delete fftw2d; delete seq1d; delete seq2d; delete gpu1d; delete gpu2d; delete [] buff;
	}

	return 0;
}

