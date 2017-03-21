# Facial_Key_Feature_Detection
In this project, OpenCV Cascade Classifier is used to locate the key features
in human face, namely eyes, nose and mouth. CUDA programming is 
used to reduce the computation time in the face detection pipeline.
Both the sequential and parallel versions of algorithms are 
implementated to compare their relative computation time.
The code is written in C++.

# Hardware requirement
1. NVIDIA GPU

# Software Requirement
1. OpenCV 2.4
2. CUDA toolkit

# How to run the program?
1. Go to src folder
2. Select OpenCV, CPU or GPU execution by uncommenting the #define option
   For instance, if I want to run only CPU program, I would uncomment 
   define CPU_EX, but comment out define GPU_EX and define OCV_EX
3. Go to build folder
	1. type cmake ..
	2. type make
	3. run the program with an image <face.img> as follows:
		./facedetector face.img  
