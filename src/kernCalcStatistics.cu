
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
