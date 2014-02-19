
__global__
void kernCalcStatistics(
	unsigned int * pHist_data,
	dim3 hist_blockDim,
	int hist_pitch,
	float * pOut
)
{
	// nothing at the moment
}

// call this with 31x31 block, 256 flat threads each. same like histogram block sizes
__global__
void kernCalcMeanMedianMaxMin(
	unsigned int * pHist_data,
	int numData, // number of data points per block (32x32 = 1024)
	/*dim3 hist_blockDim, */ // not sure if this is 31x31 or 32x32
	int hist_pitch,
	float * pOutMean,
	unsigned int * outMedian,
	unsigned int * pOutMax,
	unsigned int * pOutMin
)
{
	int hist_tid = hist_pitch * (gridDim.x * blockIdx.y + blockIdx.x);
	//TODO: copy one block of histogram to shared mem --> question: how?
	// hint: memcpy http://stackoverflow.com/questions/10456728/is-there-an-equivalent-to-memcpy-that-works-inside-a-cuda-kernel
	//__shared__ unsigned int blockHist[256] = pHist_data[ hist_tid ];
	
	//TODO: try binary reduction. atomic reduction do not give any performance gain
	
	__shared__ unsigned int blockHist[256];
	blockHist[threadIdx.x] = pHist_data[hist_tid + threadIdx.x];
	
	// mean = i * hist[i] / sum
	float mean = threadIdx.x * blockHist[threadIdx.x] / (float)numData;
	
	// output tid. output array are 31x31 matrices
	int out_tid = gridDim.x * blockIdx.y + blockIdx.x;
	
	// writing mean
	atomicAdd(&(pOutMean[ out_tid ]), mean);
	__syncthreads();
	
	//TODO: no median yet!
	
	// max & min
	if (blockHist[threadIdx.x] != 0){
		atomicMax(&(pOutMax[ out_tid ]), threadIdx.x);
		atomicMin(&(pOutMin[ out_tid ]), threadIdx.x);
	}
	__syncthreads();
	
}


//TODO: try this binary reduction https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
__global__
void kernCalcMeanMedianMaxMinBinary(
	unsigned int * pHist_data,
	int hist_pitch,
	float * pOutMean,
	unsigned int * pOutMax,
	unsigned int * pOutMin
)
{
	int thread2;
	double temp;
	//TODO: continue!
}

__global__
void kernCalcCentralMoments(
	unsigned int * pHist_data,
	int hist_pitch,
	int numData,
	float * pMean,
	float * pOutMoments2,
	float * pOutMoments3,
	float * pOutMoments4,
	float * pOutMoments5
)
{
	// temporary hist for local block
	__shared__ unsigned int blockHist[256];
	int hist_tid = hist_pitch * (gridDim.x * blockIdx.y + blockIdx.x);
	blockHist[threadIdx.x] = pHist_data[hist_tid + threadIdx.x];
	
	// out array tid (31x31)
	int out_tid = gridDim.x * blockIdx.y + blockIdx.x;
	
	float x_minus_mean = (float)threadIdx.x - pMean[out_tid];
	float m2 = blockHist[threadIdx.x] * powf(x_minus_mean, 2) / (float)numData;
	float m3 = blockHist[threadIdx.x] * powf(x_minus_mean, 3) / (float)numData;
	float m4 = blockHist[threadIdx.x] * powf(x_minus_mean, 4) / (float)numData;
	float m5 = blockHist[threadIdx.x] * powf(x_minus_mean, 5) / (float)numData;

	atomicAdd(&(pOutMoments2[out_tid]), m2);
	atomicAdd(&(pOutMoments3[out_tid]), m3);
	atomicAdd(&(pOutMoments4[out_tid]), m4);
	atomicAdd(&(pOutMoments5[out_tid]), m5);
}

// call this with 31x31 block and 1 thread per block
__global__
void kernCalcSkewnessKurtosis(
	float * pMoments2,
	float * pMoments3,
	float * pMoments4,
	float * pOutSkewness,
	float * pOutKurtosis
)
{
	int out_tid = gridDim.x * blockIdx.y + blockIdx.x;
	pOutSkewness[out_tid] = pMoments3[out_tid] / powf(pMoments2[out_tid], 1.5);
	pOutKurtosis[out_tid] = pMoments4[out_tid] / powf(pMoments2[out_tid], 2);
}
