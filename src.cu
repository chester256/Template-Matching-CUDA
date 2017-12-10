#include "src.cuh"


float* ReadBMP(const char *bmpName, int *width, int *height)
{
	FILE *fp;
	uchar *img_raw; float *image;
	int bmpwidth, bmpheight, linebyte, npixels, i, j;

	if ((fp = fopen(bmpName, "rb")) == NULL)
	{
		printf("Failed to open the image.\n");
		return 0;
	}

	if (fseek(fp, sizeof(BITMAPFILEHEADER), 0))
	{
		printf("Failed to skip the file header.\n");
		return 0;
	}

	BITMAPINFOHEADER bmpInfoHeader;
	fread(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	bmpwidth = bmpInfoHeader.biWidth;
	bmpheight = bmpInfoHeader.biHeight;
	npixels = bmpwidth*bmpheight;
	linebyte = (bmpwidth * 24 / 8 + 3) / 4 * 4;

	img_raw = (uchar*)malloc(linebyte*bmpheight);
	fread(img_raw, linebyte*bmpheight, 1, fp);

	image = (float*)malloc(sizeof(float)*npixels);
	for (i = 0; i < bmpheight; i++)
		for (j = 0; j < bmpwidth; j++)
			image[i*bmpwidth + j] = (float)img_raw[i*linebyte + j * 3];
	*width = bmpwidth;
	*height = bmpheight;

	free(img_raw);
	fclose(fp);
	return image;
}


void MarkAndSave(const char* bmpName, int X1, int Y1, int X2, int Y2, const char* outputBmpName)
{
	FILE *fp;
	uchar *img_raw; float *image;
	BITMAPFILEHEADER bmpFileHeader;
	BITMAPINFOHEADER bmpInfoHeader;
	int bmpwidth, bmpheight, linebyte, npixels;
	if ((fp = fopen(bmpName, "rb")) == NULL)
	{
		printf("Failed to open the original image.\n");
		return;
	}

	fread(&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
	fread(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	bmpwidth = bmpInfoHeader.biWidth;
	bmpheight = bmpInfoHeader.biHeight;
	npixels = bmpwidth*bmpheight;
	linebyte = (bmpwidth * 24 / 8 + 3) / 4 * 4;

	img_raw = (uchar*)malloc(linebyte*bmpheight);
	fread(img_raw, linebyte*bmpheight, 1, fp);
	fclose(fp);

	if (X1 < 0 || X2 >= bmpwidth || Y1 < 0 || Y2 >= bmpheight)
	{
		printf("Invalid rectangle position!\n");
		return;
	}
	int i;
	for (i = X1; i <= X2; i++)
	{
		img_raw[Y1*linebyte + i * 3] = 0;
		img_raw[Y1*linebyte + i * 3 + 1] = 0;
		img_raw[Y1*linebyte + i * 3 + 2] = 255;
		img_raw[Y2*linebyte + i * 3] = 0;
		img_raw[Y2*linebyte + i * 3 + 1] = 0;
		img_raw[Y2*linebyte + i * 3 + 2] = 255;
	}
	for (i = Y1 + 1; i < Y2; i++)
	{
		img_raw[i*linebyte + X1 * 3] = 0;
		img_raw[i*linebyte + X1 * 3 + 1] = 0;
		img_raw[i*linebyte + X1 * 3 + 2] = 255;
		img_raw[i*linebyte + X2 * 3] = 0;
		img_raw[i*linebyte + X2 * 3 + 1] = 0;
		img_raw[i*linebyte + X2 * 3 + 2] = 255;
	}

	if ((fp = fopen(outputBmpName, "wb")) == NULL)
	{
		printf("Failed to open the output image.\n");
		return;
	}
	fwrite(&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(img_raw, linebyte*bmpheight, 1, fp);

	free(img_raw);
	fclose(fp);
}


void sum_row_cpu(float *img, float *l1_host, float *l2_host, float *lx_host, float *ly_host, int I_width, int I_height)
{
	int i, j;
	for (i = 0; i < I_height; i++) {
		for (j = 0; j < I_width; j++) {
			l1_host[i*I_width + j] = img[i*I_width + j];
			l2_host[i*I_width + j] = pow(img[i*I_width + j], 2);
			lx_host[i*I_width + j] = img[i*I_width + j] * j;
			ly_host[i*I_width + j] = img[i*I_width + j] * i;
			if (j > 0) {
				l1_host[i*I_width + j] = l1_host[i*I_width + j - 1] + img[i*I_width + j];
				l2_host[i*I_width + j] = l2_host[i*I_width + j - 1] + pow(img[i*I_width + j], 2);
				lx_host[i*I_width + j] = lx_host[i*I_width + j - 1] + img[i*I_width + j] * j;
				ly_host[i*I_width + j] = ly_host[i*I_width + j - 1] + img[i*I_width + j] * i;
			}
		}
	}
}

void sum_col_cpu(float *l1_host, float *l2_host, float *lx_host, float *ly_host, int I_width, int I_height)
{
	int i, j;
	for (i = 0; i < I_width; i++) {
		for (j = 0; j < I_height; j++) {
			if (j > 0) {
				l1_host[j*I_width + i] += l1_host[(j - 1)*I_width + i];
				l2_host[j*I_width + i] += l2_host[(j - 1)*I_width + i];
				lx_host[j*I_width + i] += lx_host[(j - 1)*I_width + i];
				ly_host[j*I_width + i] += ly_host[(j - 1)*I_width + i];
			}
		}
	}
}

int compare(float *array1, float *array2, int n)
{
	int i, error = 0;
	float thol = 0.00001f;
	for (i = 0; i < n; i++) {
		if ((fabsf(array1[i] - array2[i]) / array2[i])> thol) error++;

	}
	return error;
}

void printMatrix(float *A, int width, int height)
{
	int i, j, m;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++)
			printf("%f ", A[i * width + j]);
		printf("\n");
	}
	printf("\n\n");
}

void compute_template_feature_cpu(float *S1, float *S2, float *Sx, float *Sy, float *v1, float *v2, float *v3, float *v4, float *l1_host, float *l2_host, float *lx_host, float *ly_host, int K, int M, int N)
{
	int WIDTH = M - K + 1;
	int HEIGHT = N - K + 1;
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			S1[i*WIDTH + j] = l1_host[(i + K - 1)*M + j + K - 1] - l1_host[(i + K - 1)*M + j] - l1_host[i *M + j + K - 1] + l1_host[i *M + j];
			S2[i*WIDTH + j] = l2_host[(i + K - 1)*M + j + K - 1] - l2_host[(i + K - 1)*M + j] - l2_host[i *M + j + K - 1] + l2_host[i *M + j];
			Sx[i*WIDTH + j] = lx_host[(i + K - 1)*M + j + K - 1] - lx_host[(i + K - 1)*M + j] - lx_host[i *M + j + K - 1] + lx_host[i *M + j];
			Sy[i*WIDTH + j] = ly_host[(i + K - 1)*M + j + K - 1] - ly_host[(i + K - 1)*M + j] - ly_host[i *M + j + K - 1] + ly_host[i *M + j];
		}
	}
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			v1[i*WIDTH + j] = S1[i*WIDTH + j] / K / K;
			v2[i*WIDTH + j] = S2[i*WIDTH + j] / K / K - v1[i*WIDTH + j] * v1[i*WIDTH + j];
			v3[i*WIDTH + j] = 4.0 * (Sx[i*WIDTH + j] - (j + 1.0 * (K - 1) / 2) * S1[i*WIDTH + j]) / K / K / K;
			v4[i*WIDTH + j] = 4.0 * (Sy[i*WIDTH + j] - (i + 1.0 * (K - 1) / 2) * S1[i*WIDTH + j]) / K / K / K;
		}
	}
}

void find_min(float *X, int &X1, int &X2, int &Y1, int &Y2, int I_width, int I_height, int T_width)
{
	int idx, idx_x, idx_y;
	float min = X[0];
	for (int i = 0; i < (I_width - T_width + 1)*(I_height - T_width + 1); i++)
	{
		if (min >= X[i])
		{
			idx = i;
			min = X[i];
		}
	}
	idx_x = idx % (I_width - T_width + 1);
	idx_y = idx / (I_width - T_width + 1);

	X1 = idx_x;
	X2 = X1 + T_width - 1;
	Y1 = idx_y;
	Y2 = Y1 + T_width - 1;
}