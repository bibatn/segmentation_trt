#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

__global__ void preprocessOnGpu(unsigned char* in, float *out, float coef, float mean1, float mean2, float mean3, float std1, float std2, float std3)
{
    //determine channel, height and width
    int c = blockIdx.y;
    int h = threadIdx.x;
    int w = blockIdx.x;

    int imgHeight = blockDim.x;
    int imgWidth = gridDim.x;

    //value at input index goes to output index (to convert image in channel first format)
    int out_idx = c*imgWidth*imgHeight + h*imgWidth + w;
    int in_idx = imgWidth*3*h + w*3 + (2-c); //(2-c) is needed to change channels order (opencv bgr to standard rgb required by neural net)
    out[out_idx] = (float)in[in_idx];

    //normalize image by means and stds of ImageNet
    if (c == 0){
      out[out_idx] = (out[out_idx]/coef - mean1) / std1;
    }
    else if (c == 1){
      out[out_idx] = (out[out_idx]/coef - mean2) / std2;
    }
    else{
      out[out_idx] = (out[out_idx]/coef - mean3) / std3;
    }

}


__global__ void computeOutputOnGpu(float* in, unsigned char* out, int numClasses, float threshold)
{
    //compute current index (a calling cuda thread will compute the value of the pixel at this index)
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int imgHeight = blockDim.x;
    int imgWidth = gridDim.x;

    //compute class of the pixel and the value (confidence) of this class
    unsigned char idx_max = 1; //initialize to the first class
    float val_max = in[i]; //initialize to value of class 1
    for (unsigned char j=1; j < numClasses; ++j) //iterate through all other classes
    {
      if(in[ i + imgHeight*imgWidth*j] > val_max) //update class and value at pixel i if needed
      {
        idx_max = j+1;
        val_max = in[ i + imgHeight*imgWidth*j];
      }
    }

    //if value (confidence) associated with the corresponding class is higher than threshold, output the class number
    if (val_max > threshold)
    {
      out[i] = idx_max;
    }
    else{ //otherwise output 0 (background)
      out[i] = 0;
    }
}

cudaError_t preprocessOnGpuWrapper(unsigned char* in, float *out, int imgHeight, int imgWidth, float coef, float mean1, float mean2, float mean3, float std1, float std2, float std3){
    dim3 blocksPerGrid(imgWidth, 3, 1);
    preprocessOnGpu<<<blocksPerGrid, imgHeight>>>(in, out, coef, mean1, mean2, mean3, std1, std2, std3);
    return cudaGetLastError();
}

cudaError_t computeOutputOnGpuWrapper(float* in, unsigned char* out, int imgHeight, int imgWidth, int numClasses, float threshold){
    computeOutputOnGpu<<<imgWidth, imgHeight>>>(in, out, numClasses, threshold);
    return cudaGetLastError();
}
