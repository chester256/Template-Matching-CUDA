#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "windows.h"

#define BLOCK_WIDTH 512
#define BLOCK_HEIGHT 2
#define BLOCK_WIDTH2 2
#define BLOCK_HEIGHT2 512

__global__ void sum_row(float *img, float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height);

__global__ void sum_col(float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height);