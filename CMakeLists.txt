project(segmentation_trt)

#include(../NAMIVisualGlobalParameter.cmake)
cmake_minimum_required(VERSION "${MIN_VER_CMAKE}" FATAL_ERROR)

################# PLATFORM IDENTITY #################
set(PLATFORM 0)
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL x86_64)
  set(PLATFORM 1)
  add_definitions(-DPLATFORM=1)
  message("PLATFORM: x86_64")
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64)
  set(PLATFORM 2)
  add_definitions(-DPLATFORM=2)
  message("PLATFORM: aarch64")
else()
  message("UNKNOWN PLATFORM: " ${CMAKE_HOST_SYSTEM_PROCESSOR})
  return()
endif()

################# CMake config #################
# select warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")

# Флаги для дебага/релиза
#set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

################# compiler config ##############
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX_EXTENSIONS OFF)

################# output config ################
# Путь для выходного файла.
# Для исполняемых файлов:
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/.......)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin/debug/plugins)
  add_definitions(-DDEBUG_BUILD)
else()
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin/release/plugins)
endif()

# for Qt
#if(COMMAND cmake_policy)
#  cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)
#add_definitions(-DQT_NO_VERSION_TAGGING)

################# library config ###############
# OpenCV
set(OpenCV_FOUND 1)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Qt et common
if(${PLATFORM} EQUAL 1)
#  set(QT_DIR /home/alexander/Qt5.6.3/5.6.3/gcc_64)
#  set(QT_INCLUDE_DIR ${QT_DIR}/include)
  set(COMMON_LIBS_DIR /usr/lib/x86_64-linux-gnu)
else()
#  set(QT_DIR /usr/lib/aarch64-linux-gnu)
#  set(QT_INCLUDE_DIR /usr/include/aarch64-linux-gnu/qt5)
  set(COMMON_LIBS_DIR /usr/lib/aarch64-linux-gnu)
endif()
#include_directories(${QT_INCLUDE_DIR})

message("PROJECT_NAME: " ${PROJECT_NAME})
message("PROJECT SOURCE DIR: " ${PROJECT_SOURCE_DIR})
message("OpenCV INCLUDE DIR: " ${OpenCV_INCLUDE_DIRS})


################# source config ################

# Хедеры
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../include)

# Исходники
file(GLOB SRC_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/../src/*.cpp)

################# Target definition #################
# create executable app
##add_executable(${PROJECT_NAME} ${SRC_LIST})

### OR ###

################# create static/shared library (insert SHARED before ${SRC_LIST}) #################
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

################# link connect #################

# COMMON_LIBS
link_directories(${COMMON_LIBS_DIR})

# OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Boost (for example)
target_link_libraries(${PROJECT_NAME} libboost_system.so)
target_link_libraries(${PROJECT_NAME} libboost_filesystem.so)
target_link_libraries(${PROJECT_NAME} libboost_log.so)
target_link_libraries(${PROJECT_NAME} libboost_log_setup.so)
target_link_libraries(${PROJECT_NAME} libboost_thread.so)
target_link_libraries(${PROJECT_NAME} libpthread.so)            # Потоки

# Qt
#link_directories(${QT_DIR}/lib)
#target_link_libraries(${PROJECT_NAME} -lQt5Network -lQt5Core)

