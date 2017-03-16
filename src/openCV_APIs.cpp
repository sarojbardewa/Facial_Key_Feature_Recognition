////////////////////////////////////////////////////////////////
// File name 	: openCV_APIs.cpp
// Description 	: This file contains implementation of 
//		: of image processing for Viola-Jones
//		: Cascade classifier classification using OpenCV 
//		: routines.
// Author 	: Saroj Bardewa
// Revision	: 03/14/2017 v.01
////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timer.h"
using namespace cv;
using namespace std; 

int detectFacialKeypoint(char fileName[],int x, int y, Mat &roi_image,
			Mat &originalImage, int x1, int y1, int c1);
///////////////////////////////////////////
// Convert RBG image to Grayscale
//////////////////////////////////////////
void ocv_rgb2Gray(Mat &inRGBImage, Mat &grayImage,float &runtime)
{	
  
    // Record time
    // Creat timer variables
	runtime = 0.00;
    CPUTimer cpuClk4;
    cpuClk4.Start();
    switch(inRGBImage.channels()) 
    {
    	case 3:
           	cvtColor(inRGBImage,grayImage,CV_BGR2GRAY ); // Convert the image to grayscale
		waitKey(0);
           	break;
    	default:
        	inRGBImage.copyTo(grayImage);
		waitKey(0);
        	break;
    }
   
    cpuClk4.Stop();  // Stop the timer
    runtime = cpuClk4.Runtime(); // Record the time elapsed
}

///////////////////////////////////////////
// Normalize the image-OpenCV call
//////////////////////////////////////////
void ocv_normalize(Mat &inGrayImage, Mat &normImage,float &runtime)
{	
    // Record time
    // Creat timer variables
	runtime = 0.00;
     CPUTimer cpuClk5;
    cpuClk5.Start();
    // Call the normalization API
    cv::normalize(inGrayImage, normImage, 0, 255, NORM_MINMAX, CV_8UC1);
    cpuClk5.Stop();
    runtime = cpuClk5.Runtime(); // Record the time elapsed
}
///////////////////////////////////////////////////
// OpenCV Haar Cascade Classifier
///////////////////////////////////////////////////

int detectAndDisplay(Mat &inNormImage, Mat &originalImage)
{
	Mat roi_image;  	// Declare a Mat variable
 	String face_cascade_name = "face.xml";    //Haar Cascade frontal face XML file
 	CascadeClassifier face_cascade;
  
	if(!face_cascade.load(face_cascade_name)){
		cout << face_cascade_name << " file not found" <<endl;
		return 0; // No face found
	}
	
	// Define a vector of faces
 	std::vector<Rect> faces;


  	equalizeHist(inNormImage, inNormImage);      // Equalize the image historgram

  	//-- Detect faces 
 	// Multiple faces can exist in an image
  	face_cascade.detectMultiScale(inNormImage, 	// Input image
				faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, 
				Size(10, 10) );  // Size of the detector smaller window 
   
  	//cout << "Number of faces found i.e. faces.size() = " <<faces.size() <<endl;
 
 	// Mark all the faces found in an image
 	for (int i = 0; i < faces.size(); i++ )
 	{
  	 	// Create rectangle to label individual face in an image
   	 	cv::Rect RectImage(faces[i].x,faces[i].y,faces[i].width,faces[i].height); 
		// Coordinates of the face rectangle
		 int x1 = faces[i].x;
		 int y1 = faces[i].y;
      
		//Also draw the rectable on the original image
		rectangle(originalImage,  		// Image
			  RectImage,		// Rectangle
			  Scalar(0,250,0),2);  // Color
   	roi_image = inNormImage(RectImage).clone();  // Draw a rectangle around the face found
	// Also, draw the rectangle on the original image


	//////////////////////////////////////////////////////////
	// Now, for each face found,search and locate key-regions
	// faceKeyRegions, nose and mouth.
	// Use separate xml files for these key facial regions
	///////////////////////////////////////////////////////////
 
        // Call faceKeyRegion detect function
	if(!detectFacialKeypoint((char *)"Eyes.xml",20,20,
				  	roi_image,originalImage,x1,y1,20)){
		cout <<"No eye was detected! "<<endl;	
	}
	
	if(!detectFacialKeypoint((char *)"Nose2.xml",50,50,
					roi_image,originalImage,x1,y1,80)){
		cout <<"No nose was detected! "<<endl;	
	}
	
	if(!detectFacialKeypoint((char *)"Mouth.xml",65,65,
					roi_image,originalImage,x1,y1,150)){
		cout <<"No nose was detected! "<<endl;	
	}
          
     }

 return 1;  // At least one face was found
 }

///////////////////////////////////////////////////////////////////////
// This function detects key-regions on a face
//////////////////////////////////////////////////////////////////////
int detectFacialKeypoint(char fileName[],int x, int y, Mat &roi_image,
			Mat &originalImage, int x1, int y1, int c1)
{

      // Load faceKeyRegion.xml haar cascade classifier
   	String faceKeyRegion_cascade_name = fileName;    //Haar Cascade frontal face XML file
 	CascadeClassifier faceKeyRegion_cascade;
  
	if(!faceKeyRegion_cascade.load(faceKeyRegion_cascade_name))
	{
		cout << faceKeyRegion_cascade_name << "file not found" <<endl;
		return 0; 
	}
    	
	// Create a vector for faceKeyRegions matrix
	vector<Rect> faceKeyRegions;
	
	//-- Detect faceKeyRegions 
 	// Multiple faceKeyRegions can exist in an image
      	faceKeyRegion_cascade.detectMultiScale(roi_image, // Input image, it is the face image
				faceKeyRegions, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, 
				Size(x, y) ); 
	
	for(int i = 0; i <faceKeyRegions.size();i++) {
	 // Create rectangle to label the faceKeyRegions

	  cv::Rect faceKeyRegion_Rect(faceKeyRegions[i].x+x1,faceKeyRegions[i].y+y1,
				     faceKeyRegions[i].width,faceKeyRegions[i].height);
   
	// Draw the rectangle on the original image
    	  rectangle(originalImage,          //Image
		    faceKeyRegion_Rect,	    // Rectangle
		    Scalar(c1,c1+100,250),2);
	}

return 1;
}


 






		



