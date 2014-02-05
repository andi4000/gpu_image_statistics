__global__
void kernCalcCornerBlockHist(
	unsigned char * src,
	int rows,
	int cols,
	int beginX,
	int beginY,
	unsigned int * outHist
)
{
	int tid = cols * (threadIdx.y + beginY) + (threadIdx.x + beginX);
	atomicAdd(&(outHist[src[tid]]), 1);
}
