TEMPLATE = lib

CONFIG += c++17
CONFIG += plugin    #need for .so modules in oder to prevent multi-version links
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++17
QMAKE_CXXFLAGS_WARN_ON  += -Wno-unknown-pragmas
QMAKE_CXXFLAGS_WARN_OFF += -Wunused-parameter
QMAKE_CXXFLAGS_WARN_OFF += -Wno-undefined

message("PLATFORM  IDENTITY: " $$QMAKE_HOST.arch)

###### Qt,Pylon,OpenCV directories
PLATFORM = 0
contains(QMAKE_HOST.arch, x86_64):{
    # for PC
    PLATFORM = 1
    DEFINES += PLATFORM=1
    message("PLATFORM: x86_64")
    OPENCV_INCLUDE_DIR = /usr/local/include/opencv4
    COMMON_LIBS_DIR = /usr/lib/x86_64-linux-gnu
#   QT_DIR = /home/alexander/Qt5.6.3/5.6.3/gcc_64
#   LIBS += -L$$QT_DIR/lib
}
contains(QMAKE_HOST.arch, aarch64):{
    # for Tegras
    PLATFORM = 2
    DEFINES += PLATFORM=2
    message("PLATFORM: aarch")
    OPENCV_INCLUDE_DIR = /usr/local/include/opencv
    COMMON_LIBS_DIR = /usr/lib/aarch64-linux-gnu
#   QT_DIR = /usr/lib/aarch64-linux-gnu
#   LIBS += -L$$QT_DIR
}

# Build properties

#Отключить "теневую сборку" в криейторе!
CONFIG(release, debug|release) {

message(Project $$TARGET (Release))

DESTDIR = $$PWD/../bin/release/plugins

OBJECTS_DIR = build/release
MOC_DIR = build/release
RCC_DIR = build/release
UI_DIR = build/release
}
CONFIG(debug, debug|release) {

message(Project $$TARGET (Debug))

DESTDIR = $$PWD/../bin/debug/plugins

OBJECTS_DIR = build/debug
MOC_DIR = build/debug
RCC_DIR = build/debug
UI_DIR = build/debug

DEFINES += DEBUG_BUILD
}

# Project files
INCLUDEPATH += $$PWD/include
SOURCES += \
    src/int8_calibrator.cpp \
    src/segmentation_trt.cpp \
    src/segmentator.cpp \
    src/trt_utils.cpp

HEADERS += \
    include/int8_calibrator.h \
    include/segmentation_trt.h \
    include/segmentator.h \
    include/trt_utils.h

# Common

INCLUDEPATH += $$PWD/../include
HEADERS += \
    ../include/base_module.h \
    ../include/messages.hpp \
    ../include/module_data.h \
    ../include/object_descriptor.h \
#    ../include/ocv_lib.h \
    ../include/parameters.h \
    ../include/utils.h \
    ../include/vision.h \
    ../include/logger.h \
    ../include/image.h \
    ../include/message_controller.h

SOURCES += \
    ../src/base_module.cpp \
    ../src/object_descriptor.cpp \
    ../src/parameters.cpp \
    ../src/logger.cpp \
    ../src/utils.cpp \
    ../src/image.cpp \
    ../src/message_controller.cpp

# Boost and common
LIBS += -L$$COMMON_LIBS_DIR \
        -lboost_filesystem \
        -lboost_system \
        -lboost_log \
        -lboost_log_setup \
        -lboost_thread \
        -lpthread \
        -ldl -fPIC

# Qt
#INCLUDEPATH += $$QT_DIR/include
#LIBS += -lQt5Network -lQt5Core

# OpenCV
INCLUDEPATH += $$OPENCV_INCLUDE_DIR
LIBS += -L/usr/local/lib \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lopencv_video \
        -lopencv_videoio \
        -lopencv_imgcodecs
#        -lopencv_features2d \
#        -lopencv_ml \
#        -lopencv_calib3d \
#        -lopencv_objdetect \
#        -lopencv_flann \
#        -lopencv_cudawarping \
#        -lopencv_cudaobjdetect \

# Cuda sources
CUDA_SOURCES += \
    src/kernels.cu

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
cuda.commands = /usr/local/cuda/bin/nvcc -c ${QMAKE_FILE_NAME} --compiler-options '-fPIC' -o ${QMAKE_FILE_OUT}

cuda.CONFIG += no_link
cuda.variable_out = OBJECTS

QMAKE_EXTRA_COMPILERS += cuda


#TensorRT
TENSORRT_PATH = /home/user/TensorRT-7.2.2.3
INCLUDEPATH += $$TENSORRT_PATH/include/
LIBS += -L/$$TENSORRT_PATH/lib \
        -lnvinfer \
        -lnvinfer_plugin \
        -lnvonnxparser


#Cuda
INCLUDEPATH +=/usr/local/cuda/include/
INCLUDEPATH +=/usr/local/cuda/targets/aarch64-linux/include/
LIBS += -L/usr/local/cuda/lib64 \
        -lcudart \
        -lcublas \
        -lcurand

