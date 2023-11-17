/**
 *  @company ФГУП НАМИ, Россия, Москва (NAMI, Russia), http://nami.ru/
 *  @date    02.03.2021
 *  Alexander Krapukhin and Yaroslav Ivanov
 **/

#ifndef VISION_SEGMENTATION_TRT_H
#define VISION_SEGMENTATION_TRT_H

#include <vision.h>
#include <base_module.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <numeric>
#include "segmentator.h"

namespace vision
{


/**
 * Описание модуля
 */
class SegmentationTrt: public BaseModule
{
public:

    /**
     * Конструктор модуля с указанием ID и типа.
     * @param id Уникальный ID экземпляра
     */
    SegmentationTrt(unsigned int /*in*/id, PLogger& logger, Parameters* global, MessageController* cntrl);

    /**
     * Деструктор модуля
     */
    ~SegmentationTrt();

protected:
    Segmentator segmentator_;

    //crop mask from bottom (value in pixels)
    int maskCrop_ = 0;

    //print fps to terminal
    bool showFPS_ = false;

    //Precision of the neural network (fp32, fp16, int8)
    std::string netprec_ = "fp32";

    //vars for out_ethernet
    uint32_t messageId_ = 0;
    string portList_;
    string ipList_;


    /**
     * @param params Входные параметры для инициализации модуля
     * @return result
     */
    VStatus onInit(const Parameters& params);

    /**
     * Обработка входных данных от предыдущих модулей
     * @param data Данные для обработки.
     * @param output Выходные данные.
     * @return result
     */
    VStatus onMessage(ModuleData* /*in*/data, ModuleData* output);


    /**
     * Проверка набора входных данных. Вызывается ПЕРЕД onMessage().
     * @param data Данные для проверки
     * @return result
     */
    VStatus validateMessage(const ModuleData& /*in*/data);


    /// Перечисление ожидаемых и обязательных параметров модуля
    EXPECTED_START
        PARAMETER("modelPath", dtString, true, "");
        PARAMETER("height", dtInt, true, 512);
        PARAMETER("width", dtInt, true, 1024);
        PARAMETER("enginePath", dtString, false, "");
        PARAMETER("precision", dtString, true, "fp32");
        PARAMETER("tablePath", dtString, false, "");
        PARAMETER("calib_images", dtString, false, "");
        PARAMETER("maskCrop", dtInt, false, 0);
        PARAMETER("threshold", dtDouble, false, 0.5);
        PARAMETER("fps", dtBool, false, false);
        PARAMETER("messageId", dtInt, true, 5558);
        PARAMETER("portList", dtString, true, "17103");
        PARAMETER("ipList", dtString, true, "192.168.3.101");

        PARAMETER("std1", dtDouble, true, 1.);
        PARAMETER("std2", dtDouble, true, 1.);
        PARAMETER("std3", dtDouble, true, 1.);
        PARAMETER("mean1", dtDouble, true, 0.);
        PARAMETER("mean2", dtDouble, true, 0.);
        PARAMETER("mean3", dtDouble, true, 0.);
        PARAMETER("coef", dtDouble, true, 1.);

        PARAMETER("up_crop", dtInt, false, 0);
        PARAMETER("start_channel", dtInt, false, 1);
    EXPECTED_END

private:
    //if the image is from a ZED camera, specific preprocessing is made
    void zedPreprocessing(cv::Mat& img);
};

/**
 * Экспортируемая функция создания модуля
 */
extern "C" __attribute__((visibility("default"))) BaseModule* create(const uint& id, PLogger* logger, Parameters* global, MessageController* cntrl);

} // end of namespace vision

#endif //#ifndef VISION_SEGMENTATION_TRT_H
