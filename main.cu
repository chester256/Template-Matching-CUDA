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
	////int x1, y1, x2, y2;

	////set the file location of I, T, and Output
	//char I_path[] = "C:\\Users\\115010256\\Documents\\lena.bmp";
	//char T_path[] = "C:\\Users\\115010256\\Documents\\lena_t.bmp";
	//char out_path[] = "output.bmp";

	//I = ReadBMP(I_path, &I_width, &I_height);
	//T = ReadBMP(T_path, &T_width, &T_height);
	
	I_width = 500;
	I_height = 500;
	float *II;
	II = (float *)malloc(I_width * I_height * sizeof(float));

	int ii;
	for (ii = 0; ii < I_height * I_width; ii++) {
		II[ii] = 5;
	}

	//-----------------------------------------------------
	float *l1_dev, *l2_dev, *lx_dev, *ly_dev, *I_dev;
	float *l1_host, *l2_host, *lx_host, *ly_host;
	float *l1_shadow, *l2_shadow, *lx_shadow, *ly_shadow;
	size_t memsize_in;

	/*float time_gpu, time_gpu_sum = 0;
	cudaEvent_t start_gpu, end_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&end_gpu);*/

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
	//
	cudaMemcpy(I_dev, II, memsize_in, cudaMemcpyHostToDevice);
	//
	//		// Set the size of block
	//		/*if (I_width <= MAX_THREADS) {
	//			const int BLOCK_WIDTH = I_width;
	//		}
	//		else {
	//			const int BLOCK_WIDTH = MAX_THREADS;
	//		}
	//		const int BLOCK_HEIGHT = MAX_THREADS / BLOCK_WIDTH;
	//		*/
	//
	int nblock_h = I_height / BLOCK_HEIGHT + (I_height%BLOCK_HEIGHT >0);
	int nblock_w = 1;
	
	printf("block height is %d\n", nblock_h);

	dim3 nblocks(nblock_w, nblock_h);
	dim3 nthreads(BLOCK_WIDTH, BLOCK_HEIGHT);

	sum_row <<<nblocks, nthreads >>>(I_dev, l1_dev, l2_dev, lx_dev, ly_dev, I_width, I_height);
	//
	//		// Compare the results with cpu version
	sum_row_cpu(II, l1_host, l2_host, lx_host, ly_host, I_width, I_height);
	/*printf("l1_host is \n");
	printMatrix(l1_host, I_width, I_height);
	*/
	//printf("l2_host is \n");
	//printMatrix(l2_host, I_width, I_height);
	////

	//printf("lx_host is \n");
	//printMatrix(lx_host, I_width, I_height);

	//printf("ly_host is \n");
	//printMatrix(ly_host, I_width, I_height);

	cudaMemcpy(l1_shadow, l1_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(l2_shadow, l2_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(lx_shadow, lx_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(ly_shadow, ly_dev, memsize_in, cudaMemcpyDeviceToHost);
	
	//		
	//compare(l1_shadow, l1_host, I_width * I_height);
	printf("Result of sum row, l1: %d\n", compare(l1_shadow, l1_host, I_width * I_height));
	printf("Result of sum row, l2: %d\n", compare(l2_shadow, l2_host, I_width * I_height));
	printf("Result of sum row, lx: %d\n", compare(lx_shadow, lx_host, I_width * I_height));
	printf("Result of sum row, ly: %d\n", compare(ly_shadow, ly_host, I_width * I_height));
	printf("\n");

	/*if (i_height <= max_threads) {
		int block_height = i_height;
	}
	else {
		int block_height = max_threads;
	}
	int block_width = max_threads / block_height;
	*/
	
	// Sum column -------------------------
	
	int nblock_h_col = 1;
	int nblock_w_col = I_width / BLOCK_WIDTH2 + (I_width%BLOCK_WIDTH2 > 0);
	
	dim3 nblocks_col(nblock_w_col, nblock_h_col);
	dim3 nthreads_col(BLOCK_WIDTH2, BLOCK_HEIGHT2);
	
	sum_col<<<nblocks_col, nthreads_col>>> (l1_dev, l2_dev, lx_dev, ly_dev, I_width, I_height);
	
	sum_col_cpu(l1_host, l2_host, lx_host, ly_host, I_width, I_height);
	
	/*printf("l1_host is \n");
	printMatrix(l1_host, I_width, I_height);*/

	/*printf("l2_host is \n");
	printMatrix(l2_host, I_width, I_height);

	printf("lx_host is \n");
	printMatrix(lx_host, I_width, I_height);

	printf("ly_host is \n");
	printMatrix(ly_host, I_width, I_height);*/

	cudaMemcpy(l1_shadow, l1_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(l2_shadow, l2_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(lx_shadow, lx_dev, memsize_in, cudaMemcpyDeviceToHost);
	cudaMemcpy(ly_shadow, ly_dev, memsize_in, cudaMemcpyDeviceToHost);

	/*printf("l1_shadow is \n");
	printMatrix(l1_shadow, I_width, I_height);

	printf("l2_shadow is \n");
	printMatrix(l2_shadow, I_width, I_height);

	printf("lx_shadow is \n");
	printMatrix(lx_shadow, I_width, I_height);

	printf("ly_shadow is \n");
	printMatrix(ly_shadow, I_width, I_height);*/

	/*int jj;
	for (jj = 0; jj < I_width * I_height; jj++) {
		printf("The value of device is %f, host is %f \n", lx_shadow[jj], lx_host[jj]);
	}
*/
	printf("Result of sum col, l1: %d\n", compare(l1_shadow, l1_host, I_width * I_height));
	printf("Result of sum col, l2: %d\n", compare(l2_shadow, l2_host, I_width * I_height));
	printf("Result of sum col, lx: %d\n", compare(lx_shadow, lx_host, I_width * I_height));
	printf("Result of sum col, ly: %d\n", compare(ly_shadow, ly_host, I_width * I_height));
	
	//
	//		//TO DO: perform template matching given I and T

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

	//Assuming that the best match patch is enclosed by vertices (x1,y1)(x2,y1)(x1,y2)(x2,y2)
	//MarkAndSave(I_path, x1, y1, x2, y2, out_path);
	free(II);
	//free(I); free(T);
	return 0;
}