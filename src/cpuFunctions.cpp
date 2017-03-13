///////////////////////////////////////////////////
// File name 	: cpuFunctions.cpp
// Description 	: These are cpu functions which are
//		: called from the main.cpp
// Author 	: Saroj Bardewa
// Revision	: 03/12/2017 v.01
///////////////////////////////////////////////////
#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timer.h"
using namespace cv;
using namespace std; 

///////////////////////////////////////////
// Convert RBG image to Grayscale
//////////////////////////////////////////

Mat h_rgb2Gray(Mat &inRGBImage,Mat &tempImage,float &runtime)
{	
		
	// Get the dimensions of the image
	int rows = inRGBImage.rows;
	int cols = inRGBImage.cols;
	uchar3  pixelVal;	 // Pixel intensity

	// Create temporary variables and initialize them	
	uchar3        *h_rgbImage;
	unsigned char *h_grayImage;

	h_rgbImage = (uchar3 *)inRGBImage.ptr<unsigned char>(0);
	h_grayImage = tempImage.ptr<unsigned char>(0);
	
	// Record time
	// Creat timer variables
	CPUTimer cpuClk;
	cpuClk.Start();
	
	// Loop to convert RGB to Gray
	for(int i = 0; i <rows; i++)
	{
		for(int j = 0; j <cols; j++)
		{
			pixelVal = h_rgbImage[i*rows+j];
			h_grayImage[i*rows+j] = .299f*pixelVal.z + .587f*pixelVal.y + .114f*pixelVal.x;
		}
	}	
	
	cpuClk.Stop();
	runtime = cpuClk.Runtime(); // Record the time elapsed
	
	// Write CUDA processed image
	Mat output(inRGBImage.rows, inRGBImage.cols,CV_8UC1,(void*)h_grayImage);

 	return output;
}
