#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 15
#define MASK_WIDTH_SQRD 225

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
        cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

void writePPM(PPMImage *img) {
  fprintf(stdout, "P6\n");
  fprintf(stdout, "# %s\n", COMMENT);
  fprintf(stdout, "%d %d\n", img->x, img->y);
  fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

  fwrite(img->data, 3 * img->x, img->y, stdout);
  fclose(stdout);
}

__global__ void smoothing_kernel(PPMPixel * dPixels, int * dSum, int xLimit, int yLimit, int n, int nThreads) {
  int k = nThreads * blockIdx.y + threadIdx.x;
  if (k < n) {
    int j = k/yLimit,
        i = k - j * yLimit,
        leftIndex = j - ((MASK_WIDTH - 1) / 2) - 1,
        rightIndex = j + ((MASK_WIDTH - 1) / 2),
        topIndex = i - ((MASK_WIDTH - 1) / 2) - 1,
        bottomIndex = i + ((MASK_WIDTH - 1) / 2),
        c = blockIdx.x,
        topLeftValue = topIndex >= 0 && leftIndex >= 0 ? dSum[c * n + topIndex * xLimit + leftIndex] : 0,
        topRightValue = topIndex >= 0 ? dSum[c * n + topIndex * xLimit + min(rightIndex, xLimit - 1)] : 0,
        bottomLeftValue = leftIndex >= 0 ? dSum[c * n + min(bottomIndex, yLimit -1) * xLimit + leftIndex] : 0, 
        bottomRightValue = dSum[c * n + min(bottomIndex, yLimit -1) * xLimit + min(xLimit - 1, rightIndex)];
    (&dPixels[(i * xLimit) + j].red)[c] = (bottomRightValue - bottomLeftValue - topRightValue + topLeftValue) / MASK_WIDTH_SQRD;
  }
}

void Smoothing(PPMImage *image, PPMImage *image_copy) {
  //vars
  int xLimit = image->x,
      yLimit = image->y,
      n = xLimit * yLimit,
      j, 
      i, 
      c;
  int * sum;
  int * dSum;
  PPMPixel * dPixels;
  unsigned int bytes = sizeof(PPMPixel) * n;
  int nThreads = 1024,
      nBlocks = (n + 1023)/nThreads;
  dim3 grid(3, nBlocks);
  dim3 block (nThreads);
  //mallocs
  CUDACHECK(cudaMalloc(&dPixels, bytes));
  bytes = sizeof(int) * n * 3;
  sum = (int *) malloc(bytes);
  CUDACHECK(cudaMalloc(&dSum, bytes));
  //array sum
  for (c = 0; c < 3; ++c) {
    sum[c * n] = (&image->data[0].red)[c];
    for (j = 1; j < xLimit; j++) 
      sum[c * n + j] = sum[c * n + j - 1] + (&image->data[j].red)[c];
    for (i = 1; i < yLimit; i++) 
      sum[c * n + i * xLimit] = sum[c * n + (i - 1) * xLimit] + (&image->data[i * xLimit].red)[c];
    for (i = 1; i < yLimit; i++) 
      for (j = 1; j < xLimit; j++) 
        sum[c * n + i * xLimit + j] = sum[c * n + i * xLimit + j - 1] + sum[c * n + (i - 1) * xLimit + j] - sum[c * n + (i - 1) * xLimit + j - 1] + (&image->data[i * xLimit + j].red)[c];
  }
  //smoothing
  CUDACHECK(cudaMemcpy(dSum, sum, bytes, cudaMemcpyHostToDevice));
  smoothing_kernel<<<grid, block>>> (dPixels, dSum, xLimit, yLimit, n, nThreads);
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(image->data, dPixels, sizeof(PPMPixel) * n, cudaMemcpyDeviceToHost));
  //free mem
  CUDACHECK(cudaFree(dPixels));
  free(sum);
  CUDACHECK(cudaFree(dSum));
}

int main(int argc, char *argv[]) {
  FILE *input;
  char filename[255];
  double t;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  // Read input filename
  fscanf(input, "%s\n", filename);

  // Read input file
  PPMImage *image = readPPM(filename);
  PPMImage *image_output = readPPM(filename);

  // Call Smoothing Kernel
  t = omp_get_wtime();
  Smoothing(image_output, image);
  t = omp_get_wtime() - t;

  // Write result to stdout
  writePPM(image_output);

  // Print time to stderr
  fprintf(stderr, "%lf\n", t);

  // Cleanup
  free(image);
  free(image_output);

  return 0;
}
