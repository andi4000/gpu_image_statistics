#ifndef _CPUCALCULATIONS_H
#define _CPUCALCULATIONS_H

void cpuCalcBlockHist(const cv::Mat src, int blockSizeX, int blockSizeY, int beginX, int beginY, unsigned int * outHist);
void cpuCalcHistAll(const cv::Mat src, int blockSizeX, int blockSizeY, int totalBlockX, int totalBlockY, int strideX, int strideY, unsigned int * outHist, int hist_pitch);
void processPseudoHistogram(unsigned int * hist, int dimx, int dimy, int dimz, int pitch, int blockX, int blockY, bool show=false);

void cpuCalcMeanMedianMaxMin(unsigned int * hist_data, int hist_dimx, int hist_dimy, int hist_pitch, float * outMean, unsigned int * outMedian, unsigned int * outMax, unsigned int * outMin);
void getMeanMedian(unsigned int * hist, int max, float * outMean, unsigned int * outMedian, bool show=false);

unsigned int getMax(unsigned int * hist, int max);
unsigned int getMin(unsigned int * hist, int max);

void cpuCalcVariance(unsigned int * hist_data, int hist_dimx, int hist_dimy, int hist_pitch, float * mean, int statArrayPitch, int numData, float * outVariance);

#endif
