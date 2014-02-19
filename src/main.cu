/**
 * //TODO:
 * General:
 * - need to split this into cpp, h, cu
 * - cpp and h compiled with g++, and nvcc for cu files
 * - how to separate compilation in cmake and then later link?
 * - implement same function in cpu, then compare
 * - see if Thrust library can help 
 * - https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu
 * 
 * Specific:
 * - using unsigned int for histogram data yields 1MB of data, which most are useless
 * - atomicAdd accepts only 32/64-bit word (int, float, unsigned int, etc)
 * - possible solution atomicAdd for short (16-bit) --> https://devtalk.nvidia.com/default/topic/495219/cuda-programming-and-performance/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/
 * 
 * Needed features:
 * - mean
 * - min
 * - max
 * - variance
 * - kurtosis
 * - skewness
 * - central moment 1st to 5th order
 * 
 */
 
#include "common.h"

#include "kernCalcCornerBlockHist.cu"
#include "kernCalcBlockHist.cu"
#include "kernCalcStatistics.cu"
#include "cpuCalculations.h"

using namespace std;
using namespace cv;

// Ref: http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char * file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int main (int argc, char** argv){
	Mat matSrc;
	
	if (argc == 2){
		matSrc = imread(argv[1], 0);
	} else {
		printf("Usage: %s [image file]!\n", argv[0]);
		return -1;
	}
	
	// which block to show result for testing purpose
	int tmp_whichBlockX = 30;
	int tmp_whichBlockY = 30; // referring to gpu block 0-30
	
	// image block sizes
	int imgBlockSizeX = 32, imgBlockSizeY = 32;
		
	// stride for block processing overlap
	int strideX = 16, strideY = 16;
	
	// total blocks for cuda
	int gpuBlockTotalX = matSrc.cols / strideX - 1;
	int gpuBlockTotalY = matSrc.rows / strideY - 1;
	
	// cuda grid and thread
	dim3 blocksPerGrid;
	dim3 threadsPerBlock;
	
	// cuda timers
	cudaEvent_t start, stop;
	float time_cpuHist, time_gpuHist, time_cpuStatCalc, time_gpuStatCalc;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// image data
	unsigned char * host_image = matSrc.data;
	
	// device image
	unsigned char * dev_image;
	size_t size = matSrc.rows * matSrc.cols * sizeof(unsigned char);

	
	// ================================ CPU Histogram =================================
	
	int cpuHistPitch = 256;
	unsigned int cpuHist[gpuBlockTotalX * gpuBlockTotalY * cpuHistPitch];
	memset(cpuHist, 0, gpuBlockTotalX * gpuBlockTotalY * cpuHistPitch * sizeof(unsigned int));
		
	blocksPerGrid = dim3(1,1,1);
	threadsPerBlock = dim3(imgBlockSizeX, imgBlockSizeY, 1);
	
	printf("\n============= CPU =============\n");
	
	cudaEventRecord(start, 0);
	cpuCalcHistAll(matSrc, imgBlockSizeX, imgBlockSizeY, gpuBlockTotalX, gpuBlockTotalY, strideX, strideY, cpuHist, cpuHistPitch);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time_cpuHist, start, stop);
	printf("Whole image CPU Histogram took %.5f ms\n", time_cpuHist);
	
	printf("\nHistogram sample from CPU, block (%d,%d)\n", tmp_whichBlockX, tmp_whichBlockY);
	processPseudoHistogram(cpuHist, gpuBlockTotalX, gpuBlockTotalY, cpuHistPitch, 256, tmp_whichBlockX, tmp_whichBlockY, false);
	
	
	// =============================== CPU mean median max min  && central moments, skewness, kurtosis ====================
	
	// mean, median, max, min, serialized array
	int cpuStatPitch = gpuBlockTotalX; // 31
	float cpuStatMean[gpuBlockTotalX*gpuBlockTotalY];
	unsigned int cpuStatMedian[gpuBlockTotalX*gpuBlockTotalY];
	unsigned int cpuStatMax[gpuBlockTotalX*gpuBlockTotalY];
	unsigned int cpuStatMin[gpuBlockTotalX*gpuBlockTotalY];

	// variables to hold central moments
	float cpuStatCentralMoment2[gpuBlockTotalX*gpuBlockTotalY];
	float cpuStatCentralMoment3[gpuBlockTotalX*gpuBlockTotalY];
	float cpuStatCentralMoment4[gpuBlockTotalX*gpuBlockTotalY];
	float cpuStatCentralMoment5[gpuBlockTotalX*gpuBlockTotalY];
	memset(cpuStatCentralMoment2, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));
	memset(cpuStatCentralMoment3, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));
	memset(cpuStatCentralMoment4, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));
	memset(cpuStatCentralMoment5, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));

	// variance, skewness, and kurtosis
	float cpuStatVariance[gpuBlockTotalX*gpuBlockTotalY];
	float cpuStatSkewness[gpuBlockTotalX*gpuBlockTotalY];
	float cpuStatKurtosis[gpuBlockTotalX*gpuBlockTotalY];
	memset(cpuStatVariance, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));
	memset(cpuStatSkewness, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));
	memset(cpuStatKurtosis, 0.0, gpuBlockTotalX*gpuBlockTotalY*sizeof(float));

	// timer start
	cudaEventRecord(start, 0);
	
	// calculating mean median max min
	cpuCalcMeanMedianMaxMin(cpuHist, gpuBlockTotalX, gpuBlockTotalY, cpuHistPitch, cpuStatMean, cpuStatMedian, cpuStatMax, cpuStatMin);

	// calculating moments
	cpuCalcCentralMoments(cpuHist, gpuBlockTotalX, gpuBlockTotalY, cpuHistPitch, cpuStatMean, cpuStatPitch, (imgBlockSizeX*imgBlockSizeY), cpuStatCentralMoment2, cpuStatCentralMoment3, cpuStatCentralMoment4, cpuStatCentralMoment5);
	
	// calculating variance, skewness and kurtosis
	cpuCalcVarianceSkewnessKurtosis(cpuStatCentralMoment2, cpuStatCentralMoment3, cpuStatCentralMoment4, gpuBlockTotalX, gpuBlockTotalY, cpuStatPitch, cpuStatVariance, cpuStatSkewness, cpuStatKurtosis);

	// timer stop & calculate elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_cpuStatCalc, start, stop);
	
	printf("\nCPU statistical calculation took %.5f ms\n", time_cpuStatCalc);
	printf("for block (%d, %d)\n", tmp_whichBlockX, tmp_whichBlockY);
	printf("mean = %f\n", cpuStatMean[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("median = %d\n", cpuStatMedian[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("max = %d\n", cpuStatMax[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("min = %d\n", cpuStatMin[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	
	printf("\nCentral Moments\n");
	printf("M2 = %.3f\n", cpuStatCentralMoment2[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("M3 = %.3f\n", cpuStatCentralMoment3[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("M4 = %.3f\n", cpuStatCentralMoment4[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("M5 = %.3f\n", cpuStatCentralMoment5[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	
	printf("\n");
	printf("variance = %.3f\n", cpuStatVariance[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("skewness = %.3f\n", cpuStatSkewness[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);
	printf("kurtosis = %.3f\n", cpuStatKurtosis[cpuStatPitch*tmp_whichBlockY + tmp_whichBlockX]);

	// ================================ GPU Histogram =================================
	
	// block sizes
	blocksPerGrid = dim3(gpuBlockTotalX, gpuBlockTotalY, 1);
	threadsPerBlock = dim3(imgBlockSizeX, imgBlockSizeY, 1);
	
	// histogram, pseudo multi-dimension array
	int dev_hist2_pitch = 256;
	unsigned int host_hist2[gpuBlockTotalX*gpuBlockTotalY*dev_hist2_pitch];
	unsigned int * dev_hist2;
	size_t size_hist2 = gpuBlockTotalX * gpuBlockTotalY * dev_hist2_pitch * sizeof(unsigned int);
	
	// main show
	printf("\n\n============= GPU =============\n");
	printf("blocks per grid = (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
	printf("threads per block = (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
	
	// timer begin
	cudaEventRecord(start,0);
	
	// allocating and copying memory in device
	cudaMalloc(&dev_image, size);
	cudaMemcpy(dev_image, host_image, size, cudaMemcpyHostToDevice);	
	cudaMalloc(&dev_hist2, size_hist2);
	cudaMemset(dev_hist2, 0, size_hist2);
	
	// kernel call
	kernCalcBlockHist<<<blocksPerGrid, threadsPerBlock>>>(dev_image, matSrc.rows, matSrc.cols, strideX, strideY, dev_hist2, dev_hist2_pitch);

	// copy the result back
	cudaMemcpy(host_hist2, dev_hist2, size_hist2, cudaMemcpyDeviceToHost);
	
	// timer end
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	// print out time
	cudaEventElapsedTime(&time_gpuHist, start, stop);
	printf("Whole image GPU histogram took %.5f ms, %.2fx %s than CPU\n", time_gpuHist, (time_cpuHist/time_gpuHist), (time_gpuHist<time_cpuHist)?"faster":"slower");

	// testing result
	printf("\nHistogram sample from GPU, block (%d,%d)\n", tmp_whichBlockX, tmp_whichBlockY);
	//TODO: needed no more?
	processPseudoHistogram(host_hist2, gpuBlockTotalX, gpuBlockTotalY, dev_hist2_pitch, 256, tmp_whichBlockX, tmp_whichBlockY, false);

	// =============================== GPU mean median max min ====================
	// gpu histogram
	
	// host variables
	int statArraySize = gpuBlockTotalX*gpuBlockTotalY;
	int statArrayStride = gpuBlockTotalX;
	float host_statMean[statArraySize];
	//unsigned int host_statMedian[statArraySize];
	unsigned int host_statMax[statArraySize];
	unsigned int host_statMin[statArraySize];
	
	// device variables
	unsigned int * dev_hist2stat;
	float * dev_statMean;
	//unsigned int * dev_statMedian;
	unsigned int * dev_statMax;
	unsigned int * dev_statMin;
	
	printf("\nGPU mean median max min\n");
	printf("blocks per grid = (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
	printf("threads per block = (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
	
	// timer start
	cudaEventRecord(start, 0);
	
	// device memory allocation
	gpuErrChk( cudaMalloc(&dev_hist2stat, size_hist2) );
	gpuErrChk( cudaMalloc(&dev_statMean, statArraySize * sizeof(float)) );
	//cudaMalloc(&dev_statMedian, statArraySize * sizeof(unsigned int));
	gpuErrChk( cudaMalloc(&dev_statMax, statArraySize * sizeof(unsigned int)) );
	gpuErrChk( cudaMalloc(&dev_statMin, statArraySize * sizeof(unsigned int)) );
	
	// copy old histogram to new
	//TODO: try cudaMemcpyHostToHost! --> seg fault
	gpuErrChk( cudaMemcpy(dev_hist2stat, host_hist2, size_hist2, cudaMemcpyHostToDevice) );
	
	// initialization
	gpuErrChk( cudaMemset(dev_statMean, 0, statArraySize * sizeof(float)) );
	//cudaMemset(dev_statMedian, 0, statArraySize * sizeof(unsigned int));
	gpuErrChk( cudaMemset(dev_statMax, 0, statArraySize * sizeof(unsigned int)) );
	gpuErrChk( cudaMemset(dev_statMin, 255, statArraySize * sizeof(unsigned int)) );
	
	cudaPrintfInit();
	
	// kernel call
	//kernCalcMeanMedianMaxMin<<<blocksPerGrid, dev_hist2_pitch>>>(dev_hist2, dev_hist2_pitch, dev_statMean, dev_statMedian, dev_statMax, dev_statMin);
	kernCalcMeanMedianMaxMin<<<blocksPerGrid, dev_hist2_pitch>>>(dev_hist2stat, dev_hist2_pitch, dev_statMean, dev_statMax, dev_statMin);
	
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	
	gpuErrChk( cudaPeekAtLastError() );
	gpuErrChk( cudaDeviceSynchronize() );
	
	// copy the results back to host
	gpuErrChk( cudaMemcpy(host_statMean, dev_statMean, statArraySize * sizeof(float), cudaMemcpyDeviceToHost) );
	//cudaMemcpy(host_statMedian, dev_statMedian, statArraySize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	gpuErrChk( cudaMemcpy(host_statMax, dev_statMax, statArraySize * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	gpuErrChk( cudaMemcpy(host_statMin, dev_statMin, statArraySize * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	
	// timer stop
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	// print out time
	cudaEventElapsedTime(&time_gpuStatCalc, start, stop);
	printf("\nGPU mean max min calculation took %.2f ms, %.2fx %s than CPU\n", time_gpuStatCalc, time_cpuStatCalc/time_gpuStatCalc, (time_gpuStatCalc<time_cpuStatCalc)?"faster":"slower");
	printf("for block (%d,%d)\n", tmp_whichBlockX, tmp_whichBlockY);
	printf("mean = %f\n", host_statMean[statArrayStride * tmp_whichBlockY + tmp_whichBlockX]);
	//printf("median = %d\n", host_statMedian[statArrayStride * tmp_whichBlockY + tmp_whichBlockX]);
	printf("max = %d\n", host_statMax[statArrayStride * tmp_whichBlockY + tmp_whichBlockX]);
	printf("min = %d\n", host_statMin[statArrayStride * tmp_whichBlockY + tmp_whichBlockX]);
	
	// testing for memcpy concept inside kernel
	/**
	int * test = new int[4];
	test[0] = 0;
	test[1] = 1;
	test[2] = 2;
	test[3] = 3;
	
	int aa[2];
	memcpy(aa, test, 2*sizeof(int));
	printf("\n aa[0] = %d\naa[1] = %d\n", aa[0], aa[1]);
	*/

	/**
	// testing cuprintf
	printf("testing cuprintf\n");
	cudaPrintfInit();
	//kernCalcBlockHist<<<dim3(2,2), dim3(2,2)>>>();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	*/

	// cleanup
	cudaFree(dev_image);
	cudaFree(dev_hist2);
	
	// gpu histogram 
	
	gpuErrChk( cudaFree(dev_hist2stat) );
	gpuErrChk( cudaFree(dev_statMean) );
	//cudaFree(dev_statMedian);
	gpuErrChk( cudaFree(dev_statMax) );
	gpuErrChk( cudaFree(dev_statMin) );
	
	gpuErrChk( cudaDeviceReset() );
	return 0;
}
