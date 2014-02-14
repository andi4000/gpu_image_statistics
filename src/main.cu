/**
 * //TODO:
 * General:
 * - need to split this into cpp, h, cu
 * - cpp and h compiled with g++, and nvcc for cu files
 * - how to separate compilation in cmake and then later link?
 * - implement same function in cpu, then compare
 * - see if Thrust library can help
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
	
	// block sizes
	int imgBlockSizeX = 32, imgBlockSizeY = 32;

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
		
	// stride for block processing overlap
	int strideX = 16, strideY = 16;
	
	// grids and thread for cuda
	int gpuBlockTotalX = matSrc.cols / strideX;
	int gpuBlockTotalY = matSrc.rows / strideY;

	
	// ================================ CPU Histogram =================================
	
	unsigned int cpuHist[gpuBlockTotalX * gpuBlockTotalY * 256];
	int cpuHistPitch = 256;
	
	blocksPerGrid = dim3(1,1,1);
	threadsPerBlock = dim3(imgBlockSizeX, imgBlockSizeY, 1);
	
	printf("\n============= CPU =============\n");
	
	cudaEventRecord(start, 0);
	cpuCalcHistAll(matSrc, imgBlockSizeX, imgBlockSizeY, strideX, strideY, cpuHist, cpuHistPitch);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time_cpuHist, start, stop);
	printf("Whole image CPU Histogram took %.5f ms\n", time_cpuHist);
	
	printf("\nHistogram sample from CPU, block (%d,%d)\n", tmp_whichBlockX, tmp_whichBlockY);
	processPseudoHistogram(cpuHist, gpuBlockTotalX, gpuBlockTotalY, cpuHistPitch, 256, tmp_whichBlockX, tmp_whichBlockY, false);

	/**
	// testing cuprintf
	printf("testing cuprintf\n");
	cudaPrintfInit();
	//kernCalcBlockHist<<<dim3(2,2), dim3(2,2)>>>();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	*/
	
	// =============================== CPU mean median max ====================
	
	// histogram dimension, to process serialized 32x32 histogram
	dim3 hist_blockDim = dim3(gpuBlockTotalX,gpuBlockTotalY,256);
	
	// mean, median, max, min, serialized array
	float cpuStatMean[gpuBlockTotalX*gpuBlockTotalY];
	unsigned int cpuStatMedian[gpuBlockTotalX*gpuBlockTotalY];
	unsigned int cpuStatMax[gpuBlockTotalX*gpuBlockTotalY];
	unsigned int cpuStatMin[gpuBlockTotalX*gpuBlockTotalY];
	int cpuStatStride = 32; //TODO: make this variable
	
	// calculating mean median max min
	cudaEventRecord(start, 0);
	cpuCalcMeanMedianMaxMin(cpuHist, hist_blockDim, cpuHistPitch, cpuStatMean, cpuStatMedian, cpuStatMax, cpuStatMin);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time_cpuStatCalc, start, stop);
	
	printf("\nCPU mean median max min calculation took %.5f ms\n", time_cpuStatCalc);
	printf("for block (%d, %d)\n", tmp_whichBlockX, tmp_whichBlockY);
	printf("mean = %f\n", cpuStatMean[cpuStatStride*tmp_whichBlockY + tmp_whichBlockX]);
	printf("median = %d\n", cpuStatMedian[cpuStatStride*tmp_whichBlockY + tmp_whichBlockX]);
	printf("max = %d\n", cpuStatMax[cpuStatStride*tmp_whichBlockY + tmp_whichBlockX]);
	printf("min = %d\n", cpuStatMin[cpuStatStride*tmp_whichBlockY + tmp_whichBlockX]);
	

	// ================================ GPU Histogram =================================
	
	// block sizes
	blocksPerGrid = dim3(gpuBlockTotalX-1, gpuBlockTotalY-1, 1);
	threadsPerBlock = dim3(imgBlockSizeX, imgBlockSizeY, 1);
	
	// histogram, pseudo multi-dimension array
	unsigned int host_hist2[gpuBlockTotalX*gpuBlockTotalY*256];
	unsigned int * dev_hist2;
	int dev_hist2_pitch = 256;
	size_t size_hist2 = gpuBlockTotalX * gpuBlockTotalY * 256 * sizeof(unsigned int);
	
	// main show
	printf("\n\n============= GPU =============\n");
	printf("blocks per grid = (%d, %d)\n", gpuBlockTotalX-1, gpuBlockTotalY-1);
	printf("threads per block = (%d, %d)\n", imgBlockSizeX, imgBlockSizeY);
	
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
	
	// cleanup
	cudaFree(dev_image);
	cudaFree(dev_hist2);
	cudaDeviceReset();
	return 0;
}
