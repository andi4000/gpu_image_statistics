#include "cuPrintf.cu"

__global__
void kernCalcBlockHist(

)
{
	cuPrintf("blockIdx.x %d blockIdx.y %d threadIdx.x %d threadIdx.y %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}
