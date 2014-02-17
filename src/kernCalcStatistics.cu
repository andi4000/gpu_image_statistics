
__global__
void kernCalcStatistics(
	unsigned int * hist_data,
	dim3 hist_blockDim,
	int hist_pitch,
	float * out
)
{
	// nothing at the moment
}

// call this with 31x31 block, 256 flat threads each. same like histogram block sizes
__global__
void kernCalcMeanMedianMaxMin(
	unsigned int * hist_data,
	/*int * hist_data_n,*/ // number of data points
	/*dim3 hist_blockDim, */ // not sure if this is 31x31 or 32x32
	int hist_pitch,
	float * outMean,
	/*unsigned int * outMedian,*/
	unsigned int * outMax,
	unsigned int * outMin
)
{
	int hist_tid = hist_pitch * (gridDim.x * blockIdx.y + blockIdx.x);
	//TODO: copy one block of histogram to shared mem --> question: how?
	// hint: memcpy http://stackoverflow.com/questions/10456728/is-there-an-equivalent-to-memcpy-that-works-inside-a-cuda-kernel
	//__shared__ unsigned int blockHist[256] = hist_data[ hist_tid ];
	
	//TODO: use atomicMin and atomicMax, only for 256 elements
	
	//TODO: try this
	//__shared__ unsigned int blockHist[256];
	//blockHist[threadIdx.x] = hist_data[hist_tid + threadIdx.x];
	
	//TODO: this is made from assuming img block size is 32x32
	int sum = 1024; //32*32
	
	// mean = i * hist[i] / sum
	float mean = threadIdx.x * hist_data[hist_tid + threadIdx.x] / (float)sum;
	//cuPrintf("mean = %.5f\n", mean);
	// output tid. output array are 31x31 matrices
	int out_tid = gridDim.x * blockIdx.y + blockIdx.x;
	
	// writing mean
	atomicAdd(&(outMean[ out_tid ]), mean);
	__syncthreads();
	//TODO: no median yet!
	
	// max & min
	if (hist_data[ hist_tid + threadIdx.x ] != 0)
		atomicMax(&(outMax[ out_tid ]), threadIdx.x);
	__syncthreads();
	
	if (hist_data[ hist_tid + threadIdx.x ] != 0)
		atomicMin(&(outMin[ out_tid ]), threadIdx.x);
	__syncthreads();
	
}


/**
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
	int hist_pitch,
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
*/
