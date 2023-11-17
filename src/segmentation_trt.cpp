/** 
 *  @company ФГУП НАМИ, Россия, Москва (NAMI, Russia), http://nami.ru/
 *  @date    02.03.2021
 **/

// include associated header file
#include "segmentation_trt.h"

namespace vision
{

    /// Экспортируемая функция создания модуля
    BaseModule *create(const uint &id, PLogger *logger, Parameters *global, MessageController *controller)
    {
        return new SegmentationTrt(id, *logger, global, controller);
    }

    /// Конструктор модуля со всеми необходимыми параметрами
    SegmentationTrt::SegmentationTrt(unsigned int /*in*/ id, PLogger &logger, Parameters *global, MessageController *controller)
        : BaseModule("SegmentationTrt", id, false, logger, global, controller) /// @param №3 doGrab, по умолчанию = false, но должен быть установлен в true для получения данных извне
    {
    }

    SegmentationTrt::~SegmentationTrt()
    {
    }

    VStatus SegmentationTrt::onInit(const Parameters &params)
    {
        //parameters
        maskCrop_ = params.get<int>("maskCrop");
        showFPS_ = params.get<bool>("fps");

        //out_ethernet parameters
        messageId_ = static_cast<uint32_t>(params.get<int>("messageId"));
        portList_ = params.get<string>("portList");
        vector<std::string> ports = nutils::splitString(portList_, ",");
        for (const auto &port : ports)
        {
            if (!nutils::checkPort(port))
            {
                ERROR("Неверный порт >> " + port);
                return ST_WRONGDATA;
            }
        }
        ipList_ = params.get<string>("ipList");
        vector<std::string> ips = nutils::splitString(ipList_, ",");
        for (const auto &ip : ips)
        {
            if (!nutils::checkIP(ip))
            {
                ERROR("Неверный ip >> " + ip);
                return ST_WRONGDATA;
            }
        }

        //initialize segmentation model
        segmentator_.initialize(params);
        return ST_OK;
    }

    VStatus SegmentationTrt::validateMessage(const ModuleData & /*in*/ data)
    {
        if (data.orig_images.empty())
        {
            return ST_EMPTY;
        }
        ///Проверка, что каждый элемент вектора содержит изображение
        for (uint i = 0; i < data.orig_images.size(); ++i)
        {
            if (data.orig_images[i].clone().empty())
            {
                return ST_WRONGDATA;
            }
        }
        return ST_OK;
    }

    VStatus SegmentationTrt::onMessage(ModuleData * /*in*/ data, ModuleData *output)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat image = data->orig_images[0].getImage().clone();
        cv::Mat outputOrigImage = image.clone();

        this->zedPreprocessing(image);

        //inference
        cv::Mat masks = segmentator_.process(image);

        //image with masks on it
        cv::Mat combined = segmentator_.putMasksOnImage(masks, image);

        //if crop is required
        if (maskCrop_ > 0)
        {
            cv::Mat imageBot = image.rowRange(image.rows - maskCrop_, image.rows).clone();
            combined = combined.rowRange(0, image.rows - maskCrop_);
            cv::vconcat(combined, imageBot, combined);
        }

        //send results
        if (output->updated_images.empty())
        {
            output->updated_images.push_back(masks);
            output->updated_images.push_back(combined);
        }
        else if (output->updated_images.size() == 1)
        {
            output->updated_images[0] = masks;
            output->updated_images.push_back(combined);
        }
        else
        {
            output->updated_images[0] = masks;
            output->updated_images[1] = combined;
        }

        //send original image further
        if (output->orig_images.empty())
        {
            output->orig_images.push_back(outputOrigImage);
        }
        else
        {
            output->orig_images[0] = outputOrigImage;
        }

        //commented out (see comment on the masksDescriptor() function in segmentator.h
        //auto descriptor = segmentator_.masksDescriptor(image.cols, image.rows);
        //output->updated_images[0].copyObjects(descriptor);

        //set output parameters
        output->parameters.set("messageId", static_cast<int>(messageId_));
        output->parameters.set("portList", portList_);
        output->parameters.set("ipList", ipList_);
        output->parameters.set("module_id", static_cast<string>("segmentation_trt"));

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;

        if (showFPS_)
        {
            std::cout << "TIME: " << elapsed.count() << " FPS: " << 1 / elapsed.count() << std::endl;
        }

        return ST_OK;
    }

    void SegmentationTrt::zedPreprocessing(cv::Mat &image)
    {
        //preprocessing for images from ZED camera

        //ZED images have rgba-format
        if (image.channels() == 4)
        {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }

        //ZED left and right images come together in a single frame
        //we need to split it and take the left half
        if (image.rows * 2 < image.cols)
        {
            image = image(cv::Rect(0, 0, image.cols / 2, image.rows));
        }
    }

} // end of namespace vision
