#include "cuPrintf.cu"

// kernel with pseudo 2d array histogram. currently our best guy
__global__
void kernCalcBlockHist(
	unsigned char * pSrcImg,
	int rows,
	int cols,
	int strideX,
	int strideY,
	unsigned int * pOutHist,
	int hist_pitch
)
{
	//TODO: optimize this by using shared mem for pSrcImg (?) and hist
	int tid_x = blockIdx.x * strideX + threadIdx.x;
	int tid_y = cols * blockIdx.y * strideY + cols * threadIdx.y;
	int tid = tid_y + tid_x;
	
	int hist_id = hist_pitch * ( gridDim.x * blockIdx.y + blockIdx.x ) + pSrcImg[tid];
	atomicAdd(&(pOutHist[ hist_id ]), 1);
}

// kernel with histogram only for 1 block. works fine but not practical.
__global__
void kernCalcBlockHist(
	unsigned char * pSrcImg,
	int rows,
	int cols,
	int strideX,
	int strideY,
	unsigned int * pOutHist
)
{
	int tid_x = blockIdx.x * strideX + threadIdx.x;
	int tid_y = cols * blockIdx.y * strideY + cols * threadIdx.y;
	int tid = tid_y + tid_x;
	
	if (blockIdx.x == 30 && blockIdx.y == 30)
		atomicAdd(&(pOutHist[pSrcImg[tid]]), 1);
}

__device__
void fillHistogram()
{
	
}
