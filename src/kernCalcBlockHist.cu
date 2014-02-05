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
	if (blockIdx.x == 31 && blockIdx.y == 0)
		cuPrintf("\tblock (%d, %d)\tthread (%d, %d)\ttid (%d + %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tid_y, tid_x);
		//cuPrintf("(%d, %d)\n", tid_x, tid_y);
}
