#include "cuPrintf.cu"

__global__
void kernCalcBlockHist(
	unsigned char * src,
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
	atomicAdd(&(outHist[blockIdx.x][blockIdx.y][src[tid]]), 1);
	/**
	// testing. change block idx values to check if tid correct.
	if (blockIdx.x == 30 && blockIdx.y == 30)
		cuPrintf("\tblock (%d, %d)\tthread (%d, %d)\ttid (%d + %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tid_y, tid_x);
	*/
}
