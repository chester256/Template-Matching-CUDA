#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "windows.h"


//Function ReadBMP(): to read a GRAYSCALE image (in 24-bit BMP format)
//bmpName: input, the image file name
//width, height: output, storing the image width and height
//Output float*: a 1-D array that stores the image (row-major)
//Returns 0 if the function fails.
//
//Example usage: float *image = ReadBMP("test.bmp", &width, &height)
//Note: Put the images at the same folder with your code files;
//or you have to attach the full location of it, such as
//"c:/users/lzh/documents/cuda/cuda/test.bmp" (using slash"/" instead of backslash"\")
//and remember to free(image) at the end

typedef unsigned char uchar;

float* ReadBMP(const char *bmpName, int *width, int *height);

void MarkAndSave(const char* bmpName, int X1, int Y1, int X2, int Y2, const char* outputBmpName);

// cpu and serial version for kernel sum_row 
void sum_row_cpu(float *img, float *l1_host, float *l2_host, float *lx_host, float *ly_host, int I_width, int I_height);

// cpu and serial version for kernel sum_col
void sum_col_cpu(float *l1_host, float *l2_host, float *lx_host, float *ly_host, int I_width, int I_height);

int compare(float *array1, float *array2, int n);

void printMatrix(float *A, int width, int height);

// compute four features of the template
void compute_template_feature_cpu(float *S1, float *S2, float *Sx, float *Sy, float *v1, float *v2, float *v3, float *v4, float *l1_host, float *l2_host, float *lx_host, float *ly_host, int K, int M, int N);

// find the minimum value in the array
void find_min(float *X, int &X1, int &X2, int &Y1, int &Y2, int I_width, int I_height, int T_width);