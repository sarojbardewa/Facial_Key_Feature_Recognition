#include <iostream>
#include <stdlib.h>
#include <string>
#include "globalVar.h"
#include "timer.h"
using namespace std;

#define THREADS_PER_BLOCK_N 64
//////////////////////////////////////////////////////////////
// Cuda error define
/////////////////////////////////////////////////////////////
#define CUDA_ERROR_HANDLER(x) {if((x)!=cudaSuccess){\
			cout <<" CUDA error at" <<__FILE__<<" " <<__LINE__<<endl;\
			cout <<cudaGetErrorString(cudaGetLastError())<<endl;\
			exit(EXIT_FAILURE);}}
//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Global CUDA kernel function
///////////////////////////////////////////////////////////////////////////////
__global__ void k_Gray2Norm ( unsigned char * const d_inGrayImage,
			      unsigned char * const d_outNormImage,
			      const int numRows, const int numCols,
			      const double minVal, const double maxVal)
{
	const long pixelIdx = threadIdx.x + blockDim.x*blockIdx.x;
	if(pixelIdx < numRows*numCols)
	{
		unsigned char const pixelVal = d_inGrayImage[pixelIdx];
		// Normalize to the 0-255 grayscale
		// Equation of normalization :: 
		// In = (I-Min)*((newMax-newMin)/(Max-Min))+newMin
		d_outNormImage[pixelIdx] = (pixelVal -minVal) * ((255-0)/(maxVal - minVal)) + 0;
	}

}

// Call function to call cuda function to convert RGB image to Grayscale
void k_normalize(unsigned char*h_inGrayImage,unsigned char*tempImage, float &runtime,const int &rows,const int &cols,const double &minVal,const double &maxVal )
{
	// Create variables 	
	unsigned char 	*d_inGrayImage;
        unsigned char   *d_outNormImage;
 	const size_t 	numPixels 	= rows*cols;
	const int 	numberOfBlocks 	= 1+((numPixels -1)/THREADS_PER_BLOCK_N);
	const dim3 	blockSize(THREADS_PER_BLOCK_N,1,1); // dim3 block(nx(B),ny(G),nz(R)) 
	const dim3 	gridSize(numberOfBlocks,1,1);	
		

	
        // allocate memory on the device for both input and output
	CUDA_ERROR_HANDLER(cudaMalloc((void **)&d_inGrayImage,  sizeof(unsigned char)*numPixels));
	CUDA_ERROR_HANDLER(cudaMalloc((void **)&d_outNormImage, sizeof(unsigned char)*numPixels));
	
	// Initialize the device output Gray Image to 
	CUDA_ERROR_HANDLER(cudaMemset(d_outNormImage,0,numPixels*sizeof(unsigned char)));

	// Copy the input image to the GPU
	CUDA_ERROR_HANDLER(cudaMemcpy(d_inGrayImage, h_inGrayImage, 
		    sizeof(unsigned char)* numPixels,cudaMemcpyHostToDevice));
	
	runtime = 0.00;
	// Start GPU computation time
	GpuTimer gpuTime3;
	gpuTime3.Start();
	
	// Call the CUDA kernel
	k_Gray2Norm<<<gridSize,blockSize>>>(d_inGrayImage,d_outNormImage,
					   rows,cols,minVal, maxVal);
	
	// End GPU Computation time
	gpuTime3.Stop();

	// Synchronize the threads
	CUDA_ERROR_HANDLER(cudaDeviceSynchronize());

	// Calculate the time elapsed
	runtime = gpuTime3.Elapsed();
	
	
	// Copy back the data from device to host
	CUDA_ERROR_HANDLER(cudaMemcpy(tempImage, d_outNormImage, 
			       sizeof(unsigned char)* numPixels,cudaMemcpyDeviceToHost));

	CUDA_ERROR_HANDLER(cudaFree(d_inGrayImage));
	CUDA_ERROR_HANDLER(cudaFree(d_outNormImage));


}


