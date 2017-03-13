#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timer.h"
using namespace std;
using namespace cv;
#define THREADS_PER_BLOCK_N 64
//////////////////////////////////////////////////////////////
// Cuda error define
/////////////////////////////////////////////////////////////
#define CUDA_CALL_N(x) {if((x)!=cudaSuccess){\
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

Mat k_normalize(Mat &inGrayImage,Mat &tempImage, float &runtime)
{
	// Create variables 	
	unsigned char 	*h_inGrayImage, *d_inGrayImage;
        unsigned char 	*h_outNormImage, *d_outNormImage;
 	const size_t 	numPixels 	= inGrayImage.rows* inGrayImage.cols;
	const int 	numberOfBlocks 	= 1+((numPixels -1)/THREADS_PER_BLOCK_N);
	const dim3 	blockSize(THREADS_PER_BLOCK_N,1,1); // dim3 block(nx(B),ny(G),nz(R)) 
	const dim3 	gridSize(numberOfBlocks,1,1);	
		
	//Get the minimum and maximum pixel intensity on the image
    	double minVal, maxVal;
	minMaxLoc(inGrayImage, &minVal, &maxVal);
	
	//Point to the beginning of the input image array
	h_inGrayImage 	= inGrayImage.ptr<unsigned char>(0);
	
	// Mat temporary variable
	h_outNormImage 	= tempImage.ptr<unsigned char>(0);

        // allocate memory on the device for both input and output
	CUDA_CALL_N(cudaMalloc((void **)&d_inGrayImage,  sizeof(unsigned char)*numPixels));
	CUDA_CALL_N(cudaMalloc((void **)&d_outNormImage, sizeof(unsigned char)*numPixels));
	
	// Initialize the device output Gray Image to 
	CUDA_CALL_N(cudaMemset(d_outNormImage,0,numPixels*sizeof(unsigned char)));

	// Copy the input image to the GPU
	CUDA_CALL_N(cudaMemcpy(d_inGrayImage, h_inGrayImage, 
		    sizeof(unsigned char)* numPixels,cudaMemcpyHostToDevice));
	
	// Start GPU computation time
	GpuTimer gpuTime;
	gpuTime.Start();
	
	// Call the CUDA kernel
	k_Gray2Norm<<<gridSize,blockSize>>>(d_inGrayImage,d_outNormImage,
					   inGrayImage.rows,inGrayImage.cols,
					   minVal, maxVal);
	
	// End GPU Computation time
	gpuTime.Stop();

	// Synchronize the threads
	CUDA_CALL_N(cudaDeviceSynchronize());

	// Calculate the time elapsed
	runtime = gpuTime.Elapsed();
	

	// Copy back the data from device to host
	CUDA_CALL_N(cudaMemcpy(h_outNormImage, d_outNormImage, 
			       sizeof(unsigned char)* numPixels,cudaMemcpyDeviceToHost));
	
	// Write CUDA processed image
	Mat output(inGrayImage.rows, inGrayImage.cols,CV_8UC1,(void*)h_outNormImage);

	CUDA_CALL_N(cudaFree(d_inGrayImage));
	CUDA_CALL_N(cudaFree(d_outNormImage));
	CUDA_CALL_N(cudaDeviceReset());

	 h_outNormImage = NULL;
	 h_inGrayImage 	= NULL;
	
	
	return output;
}

