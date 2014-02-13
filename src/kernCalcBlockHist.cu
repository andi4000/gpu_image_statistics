#include "cuPrintf.cu"

// kernel with 3D histogram, doesnt work
__global__
void kernCalcBlockHist(
	unsigned char * srcImg,
	int rows,
	int cols,
	int strideX,
	int strideY,
	unsigned int *** outHist
)
{
	int tid_x = blockIdx.x * strideX + threadIdx.x;
	int tid_y = cols * blockIdx.y * strideY + cols * threadIdx.y;
	int tid = tid_y + tid_x;
	
	//TODO: this gives 0s
	atomicAdd(&(outHist[blockIdx.x][blockIdx.y][srcImg[tid]]), 1);
	/**
	// testing. change block idx values to check if tid correct.
	if (blockIdx.x == 30 && blockIdx.y == 30)
		cuPrintf("\tblock (%d, %d)\tthread (%d, %d)\ttid (%d + %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tid_y, tid_x);
	*/
}

// kernel with pseudo 2d array histogram. currently our best guy
__global__
void kernCalcBlockHist(
	unsigned char * srcImg,
	int rows,
	int cols,
	int strideX,
	int strideY,
	unsigned int * outHist,
	int hist_pitch
)
{
	//TODO: optimize this by using shared mem for srcImg (?) and hist
	int tid_x = blockIdx.x * strideX + threadIdx.x;
	int tid_y = cols * blockIdx.y * strideY + cols * threadIdx.y;
	int tid = tid_y + tid_x;
	
	int hist_id = hist_pitch * ( blockDim.x * blockIdx.y + blockIdx.x ) + srcImg[tid];
	atomicAdd(&(outHist[ hist_id ]), 1);
}

// kernel with histogram only for 1 block. works fine but not practical.
__global__
void kernCalcBlockHist(
	unsigned char * srcImg,
	int rows,
	int cols,
	int strideX,
	int strideY,
	unsigned int * outHist
)
{
	int tid_x = blockIdx.x * strideX + threadIdx.x;
	int tid_y = cols * blockIdx.y * strideY + cols * threadIdx.y;
	int tid = tid_y + tid_x;
	
	if (blockIdx.x == 30 && blockIdx.y == 30)
		atomicAdd(&(outHist[srcImg[tid]]), 1);
}

__device__
void fillHistogram()
{
	
}
