#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "windows.h"
#include "src.cuh"
#include "kernel.cuh"

int main()
{
	//Just an example here - you are free to modify them
	int I_width, I_height, T_width, T_height;
	float *I, *T;
	int x1, y1, x2, y2;

	//set the file location of I, T, and Output
	char I_path[] = "img.bmp";
	char T_path[] = "img_t.bmp";
	char out_path[] = "output.bmp";

	I = ReadBMP(I_path, &I_width, &I_height);
	T = ReadBMP(T_path, &T_width, &T_height);

	//-----------------------------------------------------
	float *l1_dev, *l2_dev, *lx_dev, *ly_dev, *I_dev;
	float *l1_host, *l2_host, *lx_host, *ly_host;
	float *l1_shadow, *l2_shadow, *lx_shadow, *ly_shadow;
	size_t memsize_in;

	memsize_in = I_height * I_width * sizeof(float);

	l1_host = (float *)malloc(memsize_in);
	l2_host = (float *)malloc(memsize_in);
	lx_host = (float *)malloc(memsize_in);
	ly_host = (float *)malloc(memsize_in);
	l1_shadow = (float *)malloc(memsize_in);
	l2_shadow = (float *)malloc(memsize_in);
	lx_shadow = (float *)malloc(memsize_in);
	ly_shadow = (float *)malloc(memsize_in);

	cudaMalloc((void **)&l1_dev, memsize_in);
	cudaMalloc((void **)&l2_dev, memsize_in);
	cudaMalloc((void **)&lx_dev, memsize_in);
	cudaMalloc((void **)&ly_dev, memsize_in);
	cudaMalloc((void **)&I_dev, memsize_in);

	cudaMemcpy(I_dev, I, memsize_in, cudaMemcpyHostToDevice);

	// Invoke kernel sum_row
	int nblock_h = I_height / BLOCK_HEIGHT + (I_height%BLOCK_HEIGHT >0);
	int nblock_w = 1;
	dim3 nblocks(nblock_w, nblock_h);
	dim3 nthreads(BLOCK_WIDTH, BLOCK_HEIGHT);

	sum_row << <nblocks, nthreads >> >(I_dev, l1_dev, l2_dev, lx_dev, ly_dev, I_width, I_height);

	// Sum column -------------------------
	// Invoke kernel sum_col
	int nblock_h_col = 1;
	int nblock_w_col = I_width / BLOCK_WIDTH2 + (I_width%BLOCK_WIDTH2 > 0);

	dim3 nblocks_col(nblock_w_col, nblock_h_col);
	dim3 nthreads_col(BLOCK_WIDTH2, BLOCK_HEIGHT2);

	sum_col << <nblocks_col, nthreads_col >> > (l1_dev, l2_dev, lx_dev, ly_dev, I_width, I_height);

	// Since the template is small, use CPU instead of CUDA kernel
	sum_row_cpu(T, l1_host, l2_host, lx_host, ly_host, T_width, T_height);
	sum_col_cpu(l1_host, l2_host, lx_host, ly_host, T_width, T_height);

	cudaMemcpy(l1_shadow, l1_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(l2_shadow, l2_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(lx_shadow, lx_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(ly_shadow, ly_dev, memsize_in, cudaMemcpyDeviceToHost);

	//Compute feature...................................

	float *v1_dev, *v2_dev, *v3_dev, *v4_dev, *X_dev;
	float S1, S2, Sx, Sy;
	float *v1_shadow, *v2_shadow, *v3_shadow, *v4_shadow, *X_shadow;
	float vt1, vt2, vt3, vt4;

	size_t memsize_in_t, memsize_output;

	int NI = I_width * I_height;
	int NO = (I_width - T_width + 1) * (I_height - T_width + 1);

	memsize_output = NO * sizeof(float);
	memsize_in_t = T_height * T_width * sizeof(float);

	v1_shadow = (float *)malloc(memsize_output);
	v2_shadow = (float *)malloc(memsize_output);
	v3_shadow = (float *)malloc(memsize_output);
	v4_shadow = (float *)malloc(memsize_output);
	X_shadow = (float *)malloc(memsize_output);

	cudaMalloc((void **)&v1_dev, memsize_output);
	cudaMalloc((void **)&v2_dev, memsize_output);
	cudaMalloc((void **)&v3_dev, memsize_output);
	cudaMalloc((void **)&v4_dev, memsize_output);
	cudaMalloc((void **)&X_dev, memsize_output);

	// Since the template is small, use CPU instead of CUDA kernel
	compute_template_feature_cpu(&S1, &S2, &Sx, &Sy, &vt1, &vt2, &vt3, &vt4, l1_host, l2_host, lx_host, ly_host, T_width, T_width, T_height);

	// Invoke kernel to compute feature vectors and square of Euclidean distance
	int nblock_w_f = (I_width - T_width + 1) / BLOCK_WIDTH3 + ((I_width - T_width + 1) % BLOCK_HEIGHT3>0);
	int nblock_h_f = (I_height - T_width + 1) / BLOCK_HEIGHT3 + ((I_height - T_width + 1) % BLOCK_HEIGHT3>0);

	dim3 nblocks_f(nblock_w_f, nblock_h_f);
	dim3 nthreads_f(BLOCK_WIDTH3, BLOCK_HEIGHT3);

	compute_feature << <nblocks_f, nthreads_f >> >(vt1, vt2, vt3, vt4, v1_dev, v2_dev, v3_dev, v4_dev, X_dev, l1_dev, l2_dev, lx_dev, ly_dev, T_width, I_width, I_height);

	cudaMemcpy(v1_shadow, v1_dev, memsize_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(v2_shadow, v2_dev, memsize_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(v3_shadow, v3_dev, memsize_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(v4_shadow, v4_dev, memsize_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(X_shadow, X_dev, memsize_output, cudaMemcpyDeviceToHost);

	// Find the coordinates of bounding box
	find_min(X_shadow, x1, x2, y1, y2, I_width, I_height, T_width);
	printf("x1 is %d, x2 is %d, y1 is %d, y2 is %d\n", x1, x2, y1, y2);

	free(v1_shadow);
	free(v2_shadow);
	free(v3_shadow);
	free(v4_shadow);
	free(X_shadow);
	cudaFree(v1_dev);
	cudaFree(v2_dev);
	cudaFree(v3_dev);
	cudaFree(v4_dev);
	cudaFree(X_dev);

	cudaFree(l1_dev);
	cudaFree(l2_dev);
	cudaFree(lx_dev);
	cudaFree(ly_dev);
	cudaFree(I_dev);
	free(l1_host);
	free(l2_host);
	free(lx_host);
	free(ly_host);
	free(l1_shadow);
	free(l2_shadow);
	free(lx_shadow);
	free(ly_shadow);

	// Assuming that the best match patch is enclosed by vertices (x1,y1)(x2,y1)(x1,y2)(x2,y2)
	MarkAndSave(I_path, x1, y1, x2, y2, out_path);
	free(I); free(T);
	return 0;
}