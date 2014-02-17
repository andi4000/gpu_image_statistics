#include "common.h"
#include "cpuCalculations.h"

using namespace cv;
using namespace std;

void cpuCalcBlockHist(const Mat src, int blockSizeX, int blockSizeY, int beginX, int beginY, unsigned int * outHist){
	unsigned char *input = (unsigned char*) src.data;
	int count = 0;
	int bin;
	for (int j = beginY; j < beginY + blockSizeY; j++){
		for (int i = beginX; i < beginX + blockSizeX; i++){
			bin = input[src.rows * j + i];

			outHist[bin]++;
			count++;
		}
	}
}

void cpuCalcHistAll(const Mat src, int blockSizeX, int blockSizeY, int totalBlockX, int totalBlockY, int strideX, int strideY, unsigned int * outHist, int hist_pitch){
	int beginX = 0, beginY = 0;
	
	for (int j = 0; j < totalBlockY; j++){
		for (int i = 0; i < totalBlockX; i++){
			beginY = j * strideY;
			beginX = i * strideX;
			int hist_id = hist_pitch * (totalBlockX * j + i);
			cpuCalcBlockHist(src, blockSizeX, blockSizeY, beginX, beginY, (outHist + hist_id));
			//printf("Processing block (%d, %d), hist_id = %d\n", i, j, hist_id);
		}
	}
}

void cpuCalcMeanMedianMaxMin(unsigned int * hist_data, dim3 hist_blockDim, int hist_pitch, float * outMean, unsigned int * outMedian, unsigned int * outMax, unsigned int * outMin){
	// outMean, outMedian, outMax, and outMin are 2D array with dimension 31x31 which corresponds to the total blocks

	unsigned int max = 0;
	unsigned int min = 255;
	
	for (int j = 0; j < hist_blockDim.y; j++){
		for (int i = 0; i < hist_blockDim.x; i++){
			int out_id = hist_blockDim.x * j + i;
			int hist_id = hist_pitch * (hist_blockDim.x * j + i);

			getMeanMedian(hist_data + hist_id, 256, outMean + out_id, outMedian + out_id);
			outMax[out_id] = getMax(hist_data + hist_id, 256);
			outMin[out_id] = getMin(hist_data + hist_id, 256);
		}
	}
}

void getMeanMedian(unsigned int * hist, int max, float * outMean, unsigned int * outMedian, bool show){
	// calculating mean
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
	*outMean = mean;
	
	// calculating median
	int nHalf = n / 2;
	int medianSum = 0;
	
	for (int i = 0; i < max; i++){
		medianSum += hist[i];
		if (medianSum >= nHalf){
			*outMedian = i;
			break;
		}
	}
}

unsigned int getMax(unsigned int * hist, int max){
	// concept: loop from 255 to 0, return key on first nonzero value
	for (unsigned int i = max - 1; i >=0; i--){
		if (hist[i] != 0)
			return i;
	}
}

unsigned int getMin(unsigned int * hist, int max){
	// concept: loop from 0 to 255, return key on first nonzero value
	for (unsigned int i = 0; i < max; i++){
		if (hist[i] != 0)
			return i;
	}
}

void processPseudoHistogram(unsigned int * hist, int dimx, int dimy, int dimz, int pitch, int blockX, int blockY, bool show){
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
