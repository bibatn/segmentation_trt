#ifndef TRT_UTILS_H
#define TRT_UTILS_H

#include <vision.h>
#include <base_module.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <numeric>

//check cuda functions for errors
#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

//tensorrt logger
class LoggerTrt : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
};

//load image list for int8 calibration
std::vector<std::string> loadImageList(const std::string filename);

//folder where the model is located (utility function to compute other useful paths like paths to engine or calibration table)
std::string getFolderPath(std::string filePath);

//name of the model (utility function to specify names of various other files like engines etc.)
std::string getModelName(std::string filePath);

#endif // TRT_UTILS_H
