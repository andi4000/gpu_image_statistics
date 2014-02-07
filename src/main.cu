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
 
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

#include "kernCalcCornerBlockHist.cu"
#include "kernCalcBlockHist.cu"
#include "kernCalcStatistics.cu"

using namespace std;
using namespace cv;

void cpuCalcBlockHist(const Mat src, int blockSizeX, int blockSizeY, int beginX, int beginY, unsigned int * outHist){
	unsigned char *input = (unsigned char*) src.data;
	int count = 0;
	int bin;
	for (int j = beginY; j < beginY + blockSizeY; j++){
		for (int i = beginX; i < beginX + blockSizeX; i++){
			bin = input[src.rows * j + i];
			//cout<<"index = "<<src.rows * j + i<<endl;
			outHist[bin]++;
			count++;
		}
	}
	cout<<endl;
	cout<<"count = "<<count<<endl;
}

void processHistogram(unsigned int * hist, int max, bool show=false){
	float mean = 0;
	float sum = 0;
	int n = 0;
	
	for (int i = 0; i < max; i++){
		if (show)
			printf("%d\t%d\n", i, hist[i]);
		sum += i * hist[i];
		n += hist[i];
	}
	mean = sum / (float)n;
	printf("sum = %f\n", sum);
	printf("n = %d\n", n);
	printf("mean = %.5f\n", mean);
}

void processPseudoHistogram(unsigned int * hist, int dimx, int dimy, int dimz, int pitch, int blockX, int blockY, bool show=false){
	float mean = 0;
	float sum = 0;
	int n = 0;

	int idx;
	for (int i = 0; i < dimz; i++){
		idx = pitch * (dimx * blockY + blockX) + i;
		if (show)
			printf("%d\t%d\n", i, hist[idx]);
		sum += i * hist[idx];
		n += hist[idx];
	}
	
	/**
	// loop through all elements is not necessary
	for (int k = blockX; k < dimx; k++){
		for (int j = blockY; j < dimy; j++){
			for (int i = 0; i < dimz; i++){
				idx = pitch * (dimx * j + k) + i;
				if (show)
					printf("%d\t%d\n", i, hist[idx]);
				sum += i * hist[idx];
				n += hist[i];
			}
		}
	}
	*/
	
	mean = sum / (float)n;
	printf("sum = %f\n", sum);
	printf("n = %d\n", n);
	printf("mean = %.5f\n", mean);
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
	
	// block sizes
	int imgBlockSizeX = 32, imgBlockSizeY = 32;
	//int beginX = 480, beginY = 480;
	int beginX = tmp_whichBlockX * 16;
	int beginY = tmp_whichBlockY * 16;

	// cuda grid and thread
	dim3 blocksPerGrid;
	dim3 threadsPerBlock;
	
	// cuda timers
	cudaEvent_t start, stop;
	float time_kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// image data
	unsigned char * host_image = matSrc.data;
	unsigned int host_hist[256] = {0};
	
	// device image
	unsigned char * dev_image;
	size_t size = matSrc.rows * matSrc.cols * sizeof(unsigned char);
		
	// stride for block processing overlap
	int strideX = 16, strideY = 16;
	
	// grids and thread for cuda
	int gpuBlockTotalX = matSrc.cols / strideX;
	int gpuBlockTotalY = matSrc.rows / strideY;
	blocksPerGrid = dim3(gpuBlockTotalX-1, gpuBlockTotalY-1, 1);
	threadsPerBlock = dim3(imgBlockSizeX, imgBlockSizeY, 1);
	
	// histogram, pseudo multi-dimension array
	unsigned int host_hist2[gpuBlockTotalX*gpuBlockTotalY*256];
	unsigned int * dev_hist2;
	int dev_hist2_pitch = 256;
	size_t size_hist2 = gpuBlockTotalX * gpuBlockTotalY * 256 * sizeof(unsigned int);
	
	// main show
	printf("=============\n");
	printf("Running the real deal\n");
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
	cudaEventElapsedTime(&time_kernel, start, stop);
	printf("Whole image GPU histogram took %.5f ms\n", time_kernel);

	// testing result
	printf("\nhistogram for block (%d,%d) from real deal\n", tmp_whichBlockX, tmp_whichBlockY);
	processPseudoHistogram(host_hist2, gpuBlockTotalX, gpuBlockTotalY, dev_hist2_pitch, 256, tmp_whichBlockX, tmp_whichBlockY, false);
	
	
	// ================================ reference block calculation =================================
	
	blocksPerGrid = dim3(1,1,1);
	threadsPerBlock = dim3(imgBlockSizeX, imgBlockSizeY, 1);
	
	printf("\n\n===========\n");
	printf("reference calculation\n");
	
	// corner histogram
	cudaEventRecord(start, 0);
	cpuCalcBlockHist(matSrc, imgBlockSizeX, imgBlockSizeY, beginX, beginY, host_hist);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time_kernel, start, stop);
	printf("CPU Histogram took %.5f ms\n", time_kernel);
	
	printf("Histogram from CPU, result from block (%d,%d)\n", tmp_whichBlockX, tmp_whichBlockY);
	processHistogram(host_hist, 256);
	
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
//	cudaFree(dev_hist);
	cudaFree(dev_hist2);
	cudaDeviceReset();
	return 0;
}
