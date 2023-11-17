#ifndef SEGMENTATOR_H
#define SEGMENTATOR_H

#include "trt_utils.h"
#include <string>
#include "utils.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <NvOnnxParser.h>
#include <vision.h>
#include <base_module.h>
#include <array>
#include <object_descriptor.h>

//Wrapper functions declarations for CUDA kernels defined in kernels.cu
/**
* @brief preprocessOnGpuWrapper normalize input image and reshape it such that it has channels-first format (required by neural net)
* @param in pointer to memory allocated for input image
* @param out pointer to memory allocated for input image in the format require by neural network (float type, channels first)
* @param imgHeight height of the neural network input
* @param imgWidth width of the neural network input
* @return cuda error code (0 if no errors)
*/
cudaError_t preprocessOnGpuWrapper(unsigned char *in, float *out, int imgHeight, int imgWidth, float coef, float mean1, float mean2, float mean3, float std1, float std2, float std3);

/**
* @brief computeOutputOnGpuWrapper compute segmentation masks from raw neural net ouput
* @param in raw output of neural net
* @param out final mask
* @param imgHeight height of the neural network input
* @param imgWidth width of the neural network input\
* @param numClasses number of classes recognizable by neural net
* @param threshold value above which a class for a pixel is considered confident
* @return cuda error code (0 if no errors)
*/
cudaError_t computeOutputOnGpuWrapper(float *in, unsigned char *out, int imgHeight, int imgWidth, int numClasses, float threshold);

class Segmentator
{
public:
    Segmentator()
    {
    }

    virtual ~Segmentator();

    /**
    * @brief initialize segmentation model
    * @param params all parameters
    */
    void initialize(const vision::Parameters &params);

    /**
    * @brief loadEngine Laad existing engine
    * @param path Path to engine file
    * @return serialized (binary) representation of the engine (deserialized later)
    */
    std::string loadEngine(std::string const &path);

    /**
    * @brief process Do inference on input image
    * @param image Image to process
    * @return predicted masks
    */
    cv::Mat process(cv::Mat image);

    /**
    * @brief maskOnImage Place the predicted masks on the input image.
    * @param image Input image
    * @param masks Predicted masks
    * @return input image with predicted masks on it
    */
    cv::Mat putMasksOnImage(cv::Mat masks, cv::Mat image);

    /**
    * @brief build a new engine if there is no existing one
    */
    void buildEngineFromOnnx();


private:
    ObjectDescriptor getObjectDescriptor(int channel, float value, int h, int w);
    ObjectID getId(int classe);
    int step(int k = 10);

    //tensorrt engine, context, logger
    nvinfer1::ICudaEngine *engine_;
    nvinfer1::IExecutionContext *context_;
    LoggerTrt gLogger_;

    //parameters
    vision::Parameters params_;

    //pointers needed to transfer data between cpu and gpu
    unsigned char *inputCpu_;      //input image on CPU
    float *outputMaskCpu_;         //output mask on CPU
    unsigned char *inputGpu_;      //input image on GPU
    float *inputCfGpu_;            //input image on GPU in the channels first format
    float *outputRawGpu_;          //raw output of neural network on GPU (logits)
    unsigned char *outputMaskGpu_; //final output on GPU (mask with a class for each pixel)
    float *buffers_[2];            //buffers needed for context_, contain pointers to GPU memory allocated for input/output

    //value above which a class computed for a pixel is considered confident
    //range: (0, 1). Closer to 1 - more confident
    float threshold_ = 0.5;

    //neural network input dimensions
    int h_ = 0;
    int w_ = 0;

    int up_crop_ = 0;
    int start_channel_ = 1; //index of the first class (by default 0 is background, so we look for classes 1:N in the output)

    //means and stds for normalization (for example, Imagenet means and stds)
    double coef_ = 255.0;
    double mean_[3];
    double std_[3];

    //colors used for vizualization
    std::array<cv::Scalar, 11> colors_;

    vector<ClassDescr> id_;

    //number of classes
    int numClasses_ = 0;

    //paths
    std::string modelPath_;  //path to neural network model
    std::string folderPath_; //path to folder with the model
    std::string modelName_;  //name of the model
    std::string enginePath_; //path to engine
    std::string tablePath_;  //path to int8 calibration table
};

#endif // SEGMENTATOR_H
