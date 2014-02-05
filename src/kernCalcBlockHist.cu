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
	cuPrintf("(%d, %d)\n", tid_x, tid_y);
	//cuPrintf("blockIdx.x %d blockIdx.y %d threadIdx.x %d threadIdx.y %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}
