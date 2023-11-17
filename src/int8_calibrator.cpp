#include "int8_calibrator.h"

Int8EntropyCalibrator::Int8EntropyCalibrator(const std::string& calibrationSetPath,
                                             const std::string& calibTableFilePath,
                                             const uint64_t& inputCount, const uint& inputH,
                                             const uint& inputW,
                                             double coef, double* mean, double* std) :
    coef_(coef),
    h_(inputH),
    w_(inputW),
    inputCount_(inputCount),
    calibTableFilePath_(calibTableFilePath),
    imageIndex_(0)
{
    mean_ = mean;
    std_ = std;
    if (!calibrationSetPath.empty()){
        imageList_ = loadImageList(calibrationSetPath);
    }

    std::random_shuffle(imageList_.begin(), imageList_.end(), [](int i) { return rand() % i; });
    NV_CUDA_CHECK(cudaMalloc(&deviceInput_, inputCount_ * sizeof(float)));
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    std::cout << "calibration counter: " << imageIndex_+1 << "/" << imageList_.size() << std::endl;
    if (imageIndex_ + 1 >= imageList_.size()) {return false;}

    //Load next image
    cv::Mat frame = cv::imread(imageList_[imageIndex_]);
    imageIndex_++;

    //preprocess image (resize and normalization). Preprocessing must be the same as for inference
    cv::resize(frame, frame, cv::Size(w_, h_));
    frame.convertTo(frame, CV_32FC3, 1.f / coef_);
    cv::subtract(frame, cv::Scalar(mean_[0], mean_[1], mean_[2]), frame, cv::noArray(), -1);
    cv::divide(frame, cv::Scalar(std_[0], std_[1], std_[2]), frame, 1, -1);

    //copy to GPU
    NV_CUDA_CHECK(cudaMemcpy(deviceInput_, frame.ptr<float>(0), inputCount_ * sizeof(float), cudaMemcpyHostToDevice));
    bindings[0] = deviceInput_;
    return true;
}


const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    calibrationCache_.clear();
    assert(!calibTableFilePath_.empty());
    std::ifstream input(calibTableFilePath_, std::ios::binary);
    input >> std::noskipws;
    if (input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibrationCache_));
    }

    length = calibrationCache_.size();
    if (length)
    {
        std::cout << "Using cached calibration table to build the engine" << std::endl;
        output = &calibrationCache_[0];
    }
    else
    {
        std::cout << "New calibration table will be created to build the engine" << std::endl;
        output = nullptr;
    }

    return output;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    assert(!calibTableFilePath_.empty());
    std::ofstream output(calibTableFilePath_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
