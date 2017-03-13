#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timer.h"
using namespace std;
using namespace cv;
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
	outputGreyImage[pointIndex] = .299f*imagePoint.z + .587f*imagePoint.y + .114f*imagePoint.x;  // OpenCV uses BGR format
}

}

// Call function to call cuda function to convert RGB image to Grayscale

Mat k_rgb2Gray(Mat &inImage, Mat &outputGrayImage,float &runtime)
{
	// Create variables 	
	uchar3        *h_rgbImage, *d_rgbImage;  
	unsigned char *h_grayImage, *d_grayImage;
	const size_t numPixels = inImage.rows* inImage.cols;
	const int numberOfBlocks = 1+((numPixels -1)/THREADS_PER_BLOCK);
	const dim3 blockSize(THREADS_PER_BLOCK,1,1); // dim3 block(nx(B),ny(G),nz(R)) 
	const dim3 gridSize(numberOfBlocks,1,1);	
	
	h_rgbImage = (uchar3 *)inImage.ptr<unsigned char>(0);
       
	// Point to the beginning of the image array
	h_grayImage = 	outputGrayImage.ptr<unsigned char>(0);


        // allocate memory on the device for both input and output
	CUDA_CALL(cudaMalloc((void **)&d_rgbImage,sizeof(uchar3)*numPixels));
	CUDA_CALL(cudaMalloc((void **)&d_grayImage, sizeof(unsigned char)*numPixels));
	// Initialize all elements in d_grayImage to zero
	CUDA_CALL(cudaMemset(d_grayImage,0,numPixels*sizeof(unsigned char)));

	// Copy the input image to the GPU
	CUDA_CALL(cudaMemcpy(d_rgbImage, h_rgbImage,sizeof(uchar3)*numPixels,cudaMemcpyHostToDevice));
	
	// Clock GPU computation time
	GpuTimer gpuTime;
	gpuTime.Start();
	k_rgb2Gray<<<gridSize,blockSize>>>(d_rgbImage,d_grayImage,inImage.rows,inImage.cols);
	gpuTime.Stop();
	CUDA_CALL(cudaDeviceSynchronize());
	runtime = gpuTime.Elapsed();
	

	// Copy back the data from device to host
	CUDA_CALL(cudaMemcpy(h_grayImage, d_grayImage, sizeof(unsigned char)* numPixels, cudaMemcpyDeviceToHost));
	
	// Write CUDA processed image
	Mat output(inImage.rows, inImage.cols,CV_8UC1,(void*)h_grayImage);

	CUDA_CALL(cudaFree(d_rgbImage));
	CUDA_CALL(cudaFree(d_grayImage));
	CUDA_CALL(cudaDeviceReset());
	
	return output;
}

