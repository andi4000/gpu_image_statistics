#ifndef _CPUCALCULATIONS_H
#define _CPUCALCULATIONS_H

void cpuCalcBlockHist(const cv::Mat src, int blockSizeX, int blockSizeY, int beginX, int beginY, unsigned int * outHist);
void processHistogram(unsigned int * hist, int max, bool show=false);
void processPseudoHistogram(unsigned int * hist, int dimx, int dimy, int dimz, int pitch, int blockX, int blockY, bool show=false);

#endif
