#include <iostream>
#include <stdlib.h>
#include <string>

#include "globalVar.h"
#include "timer.h"
using namespace std;

#define THREADS_PER_BLOCK 64
//////////////////////////////////////////////////////////////
// Cuda error define
/////////////////////////////////////////////////////////////
#define CUDA_CALL(x) {if((x)!=cudaSuccess){\
			cout <<" CUDA error at" <<__FILE__<<" " <<__LINE__<<endl;\
			cout <<cudaGetErrorString(cudaGetLastError())<<endl;\
			exit(EXIT_FAILURE);}}
//////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Global CUDA kernel function
///////////////////////////////////////////////////////////////////////////////
__global__ void k_rgb2Gray(const uchar3* const inputRGBImage,
			        unsigned char * const outputGreyImage,
				const int numRows, const int numCols){
const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;
if(pointIndex < numRows*numCols){
	uchar3 const imagePoint = inputRGBImage[pointIndex];
	outputGreyImage[pointIndex] = .299f*imagePoint.x + .587f*imagePoint.y + .114f*imagePoint.z;  // OpenCV uses BGR format
}

}

// Call function to call cuda function to convert RGB image to Grayscale
void k_rgb2Gray(unsigned char* inImage, unsigned char*outputGrayImage,float &runtime,const int &rows,const int &cols)
{
	// Create variables 	
	uchar3        *h_rgbImage, *d_rgbImage;  
	unsigned char *d_grayImage;
	const size_t numPixels = rows*cols;
	const int numberOfBlocks = 1+((numPixels -1)/THREADS_PER_BLOCK);
	const dim3 blockSize(THREADS_PER_BLOCK,1,1); // dim3 block(nx(B),ny(G),nz(R)) 
	const dim3 gridSize(numberOfBlocks,1,1);	
	
	h_rgbImage = (uchar3 *)inImage;

        // allocate memory on the device for both input and output
	CUDA_CALL(cudaMalloc((void **)&d_rgbImage,sizeof(uchar3)*numPixels));
	CUDA_CALL(cudaMalloc((void **)&d_grayImage, sizeof(unsigned char)*numPixels));
	// Initialize all elements in d_grayImage to zero
	CUDA_CALL(cudaMemset(d_grayImage,0,numPixels*sizeof(unsigned char)));

	// Copy the input image to the GPU
	CUDA_CALL(cudaMemcpy(d_rgbImage, h_rgbImage,sizeof(uchar3)*numPixels,cudaMemcpyHostToDevice));

	// Clock GPU computation time
	GpuTimer gpuTime2;
	gpuTime2.Start();
	k_rgb2Gray<<<gridSize,blockSize>>>(d_rgbImage,d_grayImage,rows,cols);
	gpuTime2.Stop();
	CUDA_CALL(cudaDeviceSynchronize());
	runtime = gpuTime2.Elapsed();
	

	// Copy back the data from device to host
	CUDA_CALL(cudaMemcpy(outputGrayImage, d_grayImage, sizeof(unsigned char)* numPixels,cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(d_rgbImage));
	CUDA_CALL(cudaFree(d_grayImage));

}

