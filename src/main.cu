/**
 * //TODO:
 * - need to split this into cpp, h, cu
 * - cpp and h compiled with g++, and nvcc for cu files
 * - how to separate compilation in cmake and then later link?
 * 
 */
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

#include "kernCalcCornerBlockHist.cu"
#include "kernCalcBlockHist.cu"

using namespace std;
using namespace cv;

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

int main (int argc, char** argv){
	Mat matSrc;
	
	if (argc == 2){
		matSrc = imread(argv[1], 0);
	} else {
		printf("define input file!\n");
		return -1;
	}
	
	// block sizes
	int blockSizeX = 32, blockSizeY = 32;
	int beginX = 0, beginY = 0;
	
	// image data
	unsigned char * host_image = matSrc.data;
	unsigned int host_hist[256] = {0};
	
	// device image
	unsigned char * dev_image;
	size_t size = matSrc.rows * matSrc.cols * sizeof(unsigned char);
	cudaMalloc(&dev_image, size);
	cudaMemcpy(dev_image, host_image, size, cudaMemcpyHostToDevice);
	
	// device histogram
	unsigned int * dev_hist;
	cudaMalloc(&dev_hist, 256 * sizeof(unsigned int));
	cudaMemset(dev_hist, 0, 256 * sizeof(unsigned int));
	
	// cuda grid and thread
	dim3 blocksPerGrid = dim3(1,1,1);
	dim3 threadsPerBlock = dim3(blockSizeX, blockSizeY, 1);
	
	// cuda timers
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// main show
	cudaEventRecord(start, 0);
	kernCalcCornerBlockHist<<<blocksPerGrid, threadsPerBlock>>>(dev_image, matSrc.rows, matSrc.cols, beginX, beginY, dev_hist);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU Histogram took %.5f ms\n", elapsedTime);
	
	cudaMemcpy(host_hist, dev_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	// printing result
	printf("Histogram from GPU\n");
	processHistogram(host_hist, 256);
	
	// testing cuprintf
	printf("testing cuprintf\n");
	cudaPrintfInit();
	kernCalcBlockHist<<<dim3(2,2), dim3(2,2)>>>();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	
	// cleanup
	cudaFree(dev_image);
	cudaFree(dev_hist);
	cudaDeviceReset();
	return 0;
}
