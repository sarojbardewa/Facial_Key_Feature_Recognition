#ifndef __GLOBAL_HEAD__
#define __GLOBAL_HEAD__
void k_rgb2Gray(unsigned char*inImage, unsigned char*outputGrayImage,float &runtime,const int &rows,const int &cols);
void k_normalize(unsigned char*h_inGrayImage,unsigned char*tempImage, float &runtime,const int &rows,const int &cols,const double &minVal,const double &maxVal);
#endif
