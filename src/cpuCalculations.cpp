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
			//cout<<"index = "<<src.rows * j + i<<endl;
			outHist[bin]++;
			count++;
		}
	}
	//printf("count = %d\n", count);
}

void cpuCalcHistAll(const Mat src, int blockSizeX, int blockSizeY, int strideX, int strideY, unsigned int * outHist, int hist_pitch){
	int totalBlockX = src.cols / strideX;
	int totalBlockY = src.rows / strideY;
	
	int beginX = 0, beginY = 0;
	
	for (int j = 0; j < totalBlockY; j++){
		for (int i = 0; i < totalBlockX; i++){
			beginY = j * strideY;
			beginX = i * strideX;
			int hist_id = hist_pitch * (totalBlockX * j + i);
			cpuCalcBlockHist(src, blockSizeX, blockSizeY, beginX, beginY, (outHist +  hist_id));
			//printf("Processing block (%d, %d)\n", i, j);
		}
	}
}

void processHistogram(unsigned int * hist, int max, bool show){
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
