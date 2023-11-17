#include "segmentator.h"
#include "segmentation_trt.h"
#include "int8_calibrator.h"

std::string Segmentator::loadEngine(string const &path)
{
    string buffer;
    ifstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream >> noskipws;
        copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}

void Segmentator::initialize(const vision::Parameters &params)
{
    //paths
    params_ = params;
    modelPath_ = params_.get<std::string>("modelPath");
    if (!nutils::fileExists(modelPath_))
    {
        std::cerr << "model file doesn't exist: " << modelPath_ << std::endl;
        exit(1);
    }
    folderPath_ = getFolderPath(modelPath_);
    modelName_ = getModelName(modelPath_);
    enginePath_ = params_.get<std::string>("enginePath");

    //threshold
    /**
    * inverse of sigmoid to avoid taking sigmoid of logits each time and reduce computation on GPU
    * sigmoid: s = 1 / (1+e^-x)
    * inverse sigmoid: x = ln(s/(1-s))
    */
    up_crop_ = params_.get<int>("up_crop");
    start_channel_ = params_.get<int>("start_channel");

    threshold_ = static_cast<float>(params_.get<double>("threshold"));
    threshold_ = log(threshold_ / (1 - threshold_));

    //if enginePath_ is not specified, set it to a default name
    if (enginePath_.empty())
    {
        enginePath_ = folderPath_ + modelName_ + "_" + params_.get<std::string>("precision") + ".engine";
    }

    //if tablePath_ is not specified, set it to a default name
    tablePath_ = params_.get<std::string>("tablePath");
    if (tablePath_.empty())
    {
        tablePath_ = folderPath_ + modelName_ + "_calib" + ".table";
    }

    //print info
    std::cout << "modelPath_: " << modelPath_ << std::endl;
    std::cout << "folderPath_: " << folderPath_ << std::endl;
    std::cout << "modelName_: " << modelName_ << std::endl;
    std::cout << "enginePath_: " << enginePath_ << std::endl;
    std::cout << "tablePath_: " << tablePath_ << std::endl;

    //load existing engine or build a new one if needed
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger_);
    if (nutils::fileExists(enginePath_))
    {
        std::cout << "using existing engine: " << enginePath_ << std::endl;
        std::string engineData = loadEngine(enginePath_);
        engine_ = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    }
    else
    {
        buildEngineFromOnnx();
    }

    //create context for executing inference using the engine
    context_ = engine_->createExecutionContext();
    numClasses_ = context_->getBindingDimensions(1).d[1];

    h_ = params_.get<int>("height");
    w_ = params_.get<int>("width");

    coef_ = params_.get<double>("coef");

    mean_[0] = params_.get<double>("mean1");
    mean_[1] = params_.get<double>("mean2");
    mean_[2] = params_.get<double>("mean3");

    std_[0] = params_.get<double>("std1");
    std_[1] = params_.get<double>("std2");
    std_[2] = params_.get<double>("std3");

    id_ = nutils::readLabels(params_.get<string>("path_config"));

    colors_ = {
        cv::Scalar(0, 128, 128), //yellow (people)
        cv::Scalar(128, 128, 0), //cyan (cars)
        cv::Scalar(64, 64, 64), //grey (bikes)
        cv::Scalar(20, 55, 85), //brown (traffic lights)
        cv::Scalar(128, 0, 128), //magenta (signs)
        cv::Scalar(0, 64, 0), //green (road)
        cv::Scalar(37, 12, 115), //bright red (moto)
        cv::Scalar(128, 0, 0), //blue (zebra)
        cv::Scalar(42, 30, 117), //red (dashed)
        cv::Scalar(4, 0, 128), //red (solid)
        cv::Scalar(0, 0, 64)}; //red (double solid)

    //pinned memory allocation for input and output
    NV_CUDA_CHECK(cudaMallocHost((void **)&inputCpu_, h_ * w_ * 3 * sizeof(unsigned char)));
    NV_CUDA_CHECK(cudaMallocHost((void **)&outputMaskCpu_, h_ * w_ * sizeof(unsigned char)));

    //GPU memory allocation
    NV_CUDA_CHECK(cudaMalloc((void **)&inputGpu_, h_ * w_ * 3 * sizeof(unsigned char)));
    NV_CUDA_CHECK(cudaMalloc((void **)&inputCfGpu_, h_ * w_ * 3 * sizeof(float)));
    NV_CUDA_CHECK(cudaMalloc((void **)&outputRawGpu_, h_ * w_ * numClasses_ * sizeof(float)));
    NV_CUDA_CHECK(cudaMalloc((void **)&outputMaskGpu_, h_ * w_ * sizeof(unsigned char)));
}

cv::Mat Segmentator::process(cv::Mat image)
{
    //resize input image to the size required by neural net
    cv::Mat image_resized(h_, w_, CV_8UC3, inputCpu_);
    if (up_crop_ != 0)
    {
        cv::Rect crop(0, up_crop_, image.cols, image.rows - up_crop_);
        cv::Mat img_crop = image(crop);
        cv::resize(img_crop, image_resized, cv::Size(w_, h_));
    }
    else
    {
        cv::resize(image, image_resized, cv::Size(w_, h_));
    }

    //transfer the image to GPU
    NV_CUDA_CHECK(cudaMemcpy(inputGpu_, inputCpu_, h_ * w_ * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    //normalize image and reshape it such that it has channels-first format (required by neural net)
    NV_CUDA_CHECK(preprocessOnGpuWrapper(inputGpu_, inputCfGpu_, h_, w_, coef_, mean_[0], mean_[1], mean_[2], std_[0], std_[1], std_[2]));

    //inference
    buffers_[0] = inputCfGpu_;
    buffers_[1] = outputRawGpu_;
    context_->enqueue(1, (void **)buffers_, 0, nullptr);


    //compute segmentation masks from raw neural net ouput
    NV_CUDA_CHECK(computeOutputOnGpuWrapper(outputRawGpu_, outputMaskGpu_, h_, w_, numClasses_, threshold_));

    //transfer result from GPU to the host
    NV_CUDA_CHECK(cudaMemcpy(outputMaskCpu_, outputMaskGpu_, h_ * w_ * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    //resize masks to the size of the input image
    cv::Mat masks(h_, w_, 0, outputMaskCpu_);
    if (up_crop_ != 0)
    {
        cv::Mat out, resize_mask;
        cv::Mat zero(up_crop_, image.cols, 0, 0.0);
        cv::resize(masks, resize_mask, cv::Size(image.cols, image.rows - up_crop_));
        cv::vconcat(zero, resize_mask, out);
        cv::resize(out, masks, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
        cv::rectangle(image, cv::Rect(0, up_crop_, image.cols, image.rows - up_crop_), cv::Scalar(0, 0, 255), 1, 4, 0);
    }
    else
    {
        cv::resize(masks, masks, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
    }

    return masks;
}

cv::Mat Segmentator::putMasksOnImage(cv::Mat masks, cv::Mat image)
{

    //masks has only 1 channel. To paint the input image, we need a mask with 3 channels
    cv::Mat masksTemp = cv::Mat::zeros(masks.rows, masks.cols, CV_8UC3);

    //set classes to different colors
    for (int i = 0; i < numClasses_; ++i){
        masksTemp.setTo(colors_[i % numClasses_], masks == i + start_channel_);
    }
    return image + masksTemp;
}

void Segmentator::buildEngineFromOnnx()
{
    nvinfer1::IBuilder *builder{nvinfer1::createInferBuilder(gLogger_)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); //we process only one image at a time
    nvinfer1::INetworkDefinition *network{builder->createNetworkV2(explicitBatch)};
    nvonnxparser::IParser *parser{nvonnxparser::createParser(*network, gLogger_)};
    nvinfer1::IBuilderConfig *config{builder->createBuilderConfig()};

    //parse ONNX
    if (!parser->parseFromFile(modelPath_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    //remove auxilary outputs used only in the training phase
    while (network->getNbOutputs() > 1)
    {
        network->unmarkOutput(*(network->getOutput(1)));
    }

    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30); //1 as unsigned long long shifted by 30 to the left (1 GB)

    //calibrator needed if int8 is used
    nvinfer1::IInt8Calibrator *calibrator;

    //fp16 or int8 mode
    if (params_.get<std::string>("precision") == "fp16")
    {
        if (builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else
        {
            std::cerr << "ERROR: your platform does not support fp16. Use fp32 or int8.\n";
            exit(vision::ST_FAIL);
        }
    }
    else if (params_.get<std::string>("precision") == "int8")
    {
        config->setAvgTimingIterations(1);
        config->setMinTimingIterations(1);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        calibrator = new Int8EntropyCalibrator(params_.get<std::string>("calib_images"), tablePath_, h_ * w_ * 3, h_, w_, coef_, mean_, std_);
        config->setInt8Calibrator(calibrator);
    }
    //else fp32 (default precision) is used

    //we have only one image in batch
    builder->setMaxBatchSize(1);

    std::cout << "building new engine: " << enginePath_ << " ..." << std::endl;
    // generate TensorRT engine optimized for the target platform
    engine_ = builder->buildEngineWithConfig(*network, *config);

    //serialize and save the engine
    nvinfer1::IHostMemory *serializedModel = engine_->serialize();
    std::ofstream p(enginePath_, std::ios::binary);
    p.write((const char *)serializedModel->data(), serializedModel->size());
    p.close();
    serializedModel->destroy();
}

ObjectID Segmentator::getId(int classe)
{
    ObjectID objId;
    objId.ID_Hi16 = static_cast<uint16_t>(id_[classe].idCortex >> 16);
    objId.ID_Low16 = static_cast<uint16_t>((id_[classe].idCortex << 16) >> 16);
    objId.ObjectID_32 = id_[classe].idCortex;
    return objId;
}

int Segmentator::step(int k)
{
    return k;
}

ObjectDescriptor Segmentator::getObjectDescriptor(int channel, float value, int h, int w)
{
    Object_Descriptor_3D odStruct;
    odStruct.ID = this->getId(channel - 1);
    odStruct.confidence = static_cast<int>(value * 100);
    // odStruct.objectName, odStruct.cameraID, odStruct.boundBox
    odStruct.coords3d = odPoint3f(w, h);

    ObjectDescriptor od(odStruct);
    return od;
}



Segmentator::~Segmentator()
{
    //free pinned host memory
    NV_CUDA_CHECK(cudaFreeHost(inputCpu_));
    NV_CUDA_CHECK(cudaFreeHost(outputMaskCpu_));

    //free gpu memory
    NV_CUDA_CHECK(cudaFree(inputGpu_));
    NV_CUDA_CHECK(cudaFree(inputCfGpu_));
    NV_CUDA_CHECK(cudaFree(outputRawGpu_));
    NV_CUDA_CHECK(cudaFree(outputMaskGpu_));

    //destroy context and engine
    if (context_)
    {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_)
    {
        engine_->destroy();
        engine_ = nullptr;
    };
}
