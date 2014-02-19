#include "common.h"
#include "cpuCalculations.h"

using namespace cv;
using namespace std;

void cpuCalcBlockHist(const Mat src, int blockSizeX, int blockSizeY, int beginX, int beginY, unsigned int * pOutHist){
	unsigned char *pInput = (unsigned char*) src.data;
	int count = 0;
	int bin;
	for (int j = beginY; j < beginY + blockSizeY; j++){
		for (int i = beginX; i < beginX + blockSizeX; i++){
			bin = pInput[src.rows * j + i];

			pOutHist[bin]++;
			count++;
		}
	}
}

void cpuCalcHistAll(const Mat src, int blockSizeX, int blockSizeY, int totalBlockX, int totalBlockY, int strideX, int strideY, unsigned int * pOutHist, int hist_pitch){
	int beginX = 0, beginY = 0;
	
	for (int j = 0; j < totalBlockY; j++){
		for (int i = 0; i < totalBlockX; i++){
			beginY = j * strideY;
			beginX = i * strideX;
			int hist_id = hist_pitch * (totalBlockX * j + i);
			cpuCalcBlockHist(src, blockSizeX, blockSizeY, beginX, beginY, (pOutHist + hist_id));
			//printf("Processing block (%d, %d), hist_id = %d\n", i, j, hist_id);
		}
	}
}

void cpuCalcMeanMedianMaxMin(unsigned int * pHist_data, int hist_dimx, int hist_dimy, int hist_pitch, float * pOutMean, unsigned int * pOutMedian, unsigned int * pOutMax, unsigned int * pOutMin){
	// pOutMean, pOutMedian, pOutMax, and pOutMin are 2D array with dimension 31x31 which corresponds to the total blocks

	unsigned int max = 0;
	unsigned int min = 255;
	
	for (int j = 0; j < hist_dimy; j++){
		for (int i = 0; i < hist_dimx; i++){
			int out_id = hist_dimx * j + i;
			int hist_id = hist_pitch * (hist_dimx * j + i);

			getMeanMedian(pHist_data + hist_id, 256, pOutMean + out_id, pOutMedian + out_id);
			pOutMax[out_id] = getMax(pHist_data + hist_id, 256);
			pOutMin[out_id] = getMin(pHist_data + hist_id, 256);
		}
	}
}

void getMeanMedian(unsigned int * hist, int max, float * pOutMean, unsigned int * pOutMedian, bool show){
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
	*pOutMean = mean;
	
	// calculating median
	int nHalf = n / 2;
	int medianSum = 0;
	
	for (int i = 0; i < max; i++){
		medianSum += hist[i];
		if (medianSum >= nHalf){
			*pOutMedian = i;
			break;
		}
	}
}

unsigned int getMax(unsigned int * pHist, int max){
	// concept: loop from 255 to 0, return key on first nonzero value
	for (unsigned int i = max - 1; i >=0; i--){
		if (pHist[i] != 0)
			return i;
	}
}

unsigned int getMin(unsigned int * pHist, int max){
	// concept: loop from 0 to 255, return key on first nonzero value
	for (unsigned int i = 0; i < max; i++){
		if (pHist[i] != 0)
			return i;
	}
}

void processPseudoHistogram(unsigned int * pHist, int dimx, int dimy, int dimz, int pitch, int blockX, int blockY, bool show){
	float mean = 0;
	float sum = 0;
	int n = 0;

	int idx;
	for (int i = 0; i < dimz; i++){
		idx = pitch * (dimx * blockY + blockX) + i;
		if (show)
			printf("%d\t%d\n", i, pHist[idx]);
		sum += i * pHist[idx];
		n += pHist[idx];
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

/**
 * pOutVariance is 31x31 pseudo matrix
 * dont forget to memset to 0s
 * numData refers to histogram data points (1024)
 */

void cpuCalcVariance(unsigned int * pHist_data, int hist_dimx, int hist_dimy, int hist_pitch, float * pMean, int statArrayPitch, int numData, float * pOutVariance){
	// formula: sum( p(x) * (x - mean)^2 / (N - 1))
	// block loop
	unsigned int * histTmp;
	for (int j = 0; j < hist_dimy; j++){
		for (int i = 0; i < hist_dimx; i++){
			// were inside a block now.
			int array_id = statArrayPitch * j + i;
			int hist_id = hist_pitch * (hist_dimx * j + i);
			histTmp = pHist_data + hist_id;
			
			// now loop through histogram values
			for (int x = 0; x < hist_pitch; x++){
				float x_minus_mean = (float)x - pMean[array_id];
				pOutVariance[array_id] += histTmp[x] * (x_minus_mean*x_minus_mean) / (float)(numData);
				//if (i == 30 && j == 30)
					//printf("%d * (%d - %.3f)^2 / (%d - 1) = %.3f\n", histTmp[x], x, mean[array_id], numData, tmpVar);
			}
		}
	}
	
}

void cpuCalcCentralMoments(unsigned int * pHist_data, int hist_dimx, int hist_dimy, int hist_pitch, float * pMean, int statArrayPitch, int numData, float * pOutMoments2, float * pOutMoments3, float * pOutMoments4, float * pOutMoments5){
	// formula: sum( (x - mean)^order ) / numData
	unsigned int * pHistTmp;
	for (int j = 0; j < hist_dimy; j++){
		for (int i = 0; i < hist_dimx; i++){
			// inside a block
			int outArray_id = statArrayPitch * j + i;
			int hist_id = hist_pitch * (hist_dimx * j + i);
			pHistTmp = pHist_data + hist_id;
			
			// now loop through histogram data
			for (int x = 0; x < hist_pitch; x++){
				float x_minus_mean = (float)x - pMean[outArray_id];
				pOutMoments2[outArray_id] += pHistTmp[x] * pow(x_minus_mean, 2) / (float)numData;
				pOutMoments3[outArray_id] += pHistTmp[x] * pow(x_minus_mean, 3) / (float)numData;
				pOutMoments4[outArray_id] += pHistTmp[x] * pow(x_minus_mean, 4) / (float)numData;
				pOutMoments5[outArray_id] += pHistTmp[x] * pow(x_minus_mean, 5) / (float)numData;
			}
		}
	}
}

void cpuCalcVarianceSkewnessKurtosis(float * pMoments2, float * pMoments3, float * pMoments4, int arrayDimX, int arrayDimY, int statArrayPitch, float * pOutVariance, float * pOutSkewness, float * pOutKurtosis){
	// variance = 2nd central moment
	// skewness = m3 / (m2^1.5)
	// kurtosis = m4 / (m2^2)
	for (int j = 0; j < arrayDimY; j++){
		for (int i = 0; i < arrayDimY; i++){
			int arrayId = statArrayPitch * j + i;
			pOutVariance[arrayId] = pMoments2[arrayId];
			pOutSkewness[arrayId] = pMoments3[arrayId] / pow(pMoments2[arrayId], 1.5);
			pOutKurtosis[arrayId] = pMoments4[arrayId] / pow(pMoments2[arrayId], 2);
		}
	}
}
