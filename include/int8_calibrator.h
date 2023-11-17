#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_H

#include <vision.h>
#include <base_module.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <numeric>
#include "segmentator.h"
#include "trt_utils.h"

//calibrator needed for int8 inference
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(const std::string& calibrationSetPath,
                          const std::string& calibTableFilePath, const uint64_t& inputCount,
                          const uint& inputH, const uint& inputW, double coef, double* mean, double* std);
    virtual ~Int8EntropyCalibrator() { NV_CUDA_CHECK(cudaFree(deviceInput_)); }

    //we always have 1 image in input
    int getBatchSize() const noexcept override { return 1; }

    //get next image for calibration. Returns false when all images have been processed
    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override;

    //try to load existing calibration table and return a pointer to it. If it fails, creates a new table (it's very slow)
    const void *readCalibrationCache(size_t &length) noexcept override;

    //save calibration table for future reuse
    void writeCalibrationCache(const void *cache, size_t length) noexcept override;

private:

    double coef_ = 1.f;
    double *mean_;
    double *std_;
    
    //neural network input dimentsions
    const uint h_;
    const uint w_;

    //height*width*channels
    const uint64_t inputCount_;

    //path to int8 calibration table
    const std::string calibTableFilePath_{nullptr};

    //image counter
    uint imageIndex_;

    //pointer to allocated gpu memory
    void* deviceInput_{nullptr};

    //paths to calibration images
    std::vector<std::string> imageList_;

    //calibration table
    std::vector<char> calibrationCache_;
};

#endif // INT8_CALIBRATOR_H
