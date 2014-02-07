
__global__
void kernCalcStatistics(
	unsigned int * hist_data,
	dim3 hist_blockDim,
	int pitch,
	float * out
)
{
	// nothing at the moment
}

__global__
void kernCalcMeanMaxMin(
	unsigned int * hist_data,
	dim3 hist_blockDim,
	int pitch,
	float * outMean,
	unsigned int * outMax,
	unsigned int * outMin
)
{
	int hist_tid = pitch * (hist_blockDim.x * blockIdx.y + blockIdx.x);
	//TODO: copy one block of histogram to shared mem --> question: how?
	// hint: memcpy http://stackoverflow.com/questions/10456728/is-there-an-equivalent-to-memcpy-that-works-inside-a-cuda-kernel
	//__shared__ unsigned int blockHist[256] = hist_data[ hist_tid ];
	
	//TODO: use atomicMin and atomicMax, only for 256 elements
}

//TODO: is sorting necessary?
__device__
void sortHistogram()
{
	
}

__device__
unsigned int getMedian()
{
	return 0;
} 

__device__
unsigned int getMin()
{
	return 0;
}

__device__
unsigned int getMax()
{
	return 0;
}

__device__
float getMean(
	unsigned int * hist_data,
	dim3 hist_blockDim,
	int pitch,
	dim3 whichBlock
)
{
	float mean = 0;
	float sum = 0;
	int n = 0;
	int idx;
	// process the array
	return mean;
}
