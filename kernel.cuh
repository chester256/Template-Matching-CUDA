#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "windows.h"

//Block size for kernel sum_row
#define BLOCK_WIDTH 512
#define BLOCK_HEIGHT 2
//Block size for kernel sum_col
#define BLOCK_WIDTH2 2
#define BLOCK_HEIGHT2 512

//Block size for kernel computer_feature
#define BLOCK_HEIGHT3 32
#define BLOCK_WIDTH3 16

/*
Input: img, img width and img height
Output: four tables with rows summed
*/
__global__ void sum_row(float *img, float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height);

/*
Input: four row summed tables, img width and img height
Output: four tables with columns summed
*/
__global__ void sum_col(float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height);

/*
Input: four row summed-area tables, img width and img height, template width = template height
Output: square of Euclidean distance table (of size (img width-template width+1) * (img height-template height+1))
*/
__global__ void compute_feature(float vt1value, float vt2value, float vt3value, float vt4value, float *v1_dev, float *v2_dev, float *v3_dev, float *v4_dev, float *X_dev, float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int K, int M, int N);