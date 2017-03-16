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

void h_rgb2Gray(Mat &inRGBImage,Mat &tempImage,float &runtime)
{	
		
	// Get the dimensions of the image
	int rows = inRGBImage.rows;
	int cols = inRGBImage.cols;
	
	// Record time
	// Creat timer variables
	CPUTimer cpuClk;
	cpuClk.Start();
	
	cv::Vec3b intensity;
	// Loop to convert RGB to Gray
	for(int i = 0; i <rows; ++i)
	{
		for(int j = 0; j < (cols-1); ++j)
		{
			intensity = inRGBImage.at<cv::Vec3b>(i,j);
			tempImage.at<uchar>(i,j) = .299f*intensity.val[0] + .587f*intensity.val[1] + .114f*intensity.val[2];
		}
	}
	
	//Check	
	//intensity = inRGBImage.at<cv::Vec3b>(0,0);
	//cout << "CPU At [0] [B,G,R] " <<.299f*intensity.val[0] <<"," <<.587f*intensity.val[1] <<"," <<.114f*intensity.val[2]<<endl;
	
	cpuClk.Stop();
	runtime = cpuClk.Runtime(); // Record the time elapsed
	
}

///////////////////////////////////////////
// Normalize the image
//////////////////////////////////////////

void h_normalize(Mat &inGrayImage,Mat &tempImage,float &runtime)
{	
		
	// Get the dimensions of the image
	int rows = inGrayImage.rows;
	int cols = inGrayImage.cols;
	double  pixelVal;	 // Pixel intensity

	//Get the minimum and maximum pixel intensity on the image
    	double minVal, maxVal;
	minMaxLoc(inGrayImage, &minVal, &maxVal);
	// Record time
	// Creat timer variables
	CPUTimer cpuClk;
	cpuClk.Start();
	
	// Loop to normalize the images
	for(int i = 0; i <rows; i++)
	{
		for(int j = 0; j <cols; j++)
		{
			pixelVal = inGrayImage.at<uchar>(i,j);
			tempImage.at<uchar>(i,j) = (uchar) (pixelVal- minVal) * ((255-0)/(maxVal - minVal)) + 0;
		}
	}	
	
	cpuClk.Stop();
	runtime = cpuClk.Runtime(); // Record the time elapsed
	
}
