# SEGMENTATION_TRT
Segmentation module

## Purpose of the module
Semantic road scene segmentation using neural network.

## Dependencies
1. CUDA = 11.1
2. Cudnn 8.0.5
3. TensorRT 7.2.1.6
4. OepnCV = 4.5.1

## Supported models
The module supports any correct segmentation model in ONNX format

## Supported precisions
FP32, FP16, INT8

##Models

###mcnet
marking recognition
example structure: struct/struct_segment_trt_marking.yml


### bisenetv2_6
6 classes:
* people
* cars
* bicycles
*traffic lights
*signs
* road

### bisenetv2_8
8 classes:
*people
* cars
* bicycles
*traffic lights
*signs
* road
*moto
*crosswalks

### bisenetv2_11
11 classes:
* people
* cars
* bicycles
*traffic lights
*signs
* road
*moto
*crosswalks
* dashed lane
*solid lane
* double solid lane

example structure: struct/struct_segment_trt.yml

## Performance
The experiments were carried out on Nvidia Quadro P620 and GeForce GTX 1660 Ti video cards. Resolution of the tensor included in the neural network: 512x1024. The measurements include the entire cycle with pre-processing and post-processing.


| Video card | FPS fp32 | FPS fp16 | FPS int8 |
| ------ | ------ | ------ | ------ |
| Quadro P620 | 19-21 | not supported | 28-32 |
| GTX 1660 Ti | 48-58 | 50-75 | 55-80 |
<br>

## Links
Bisenet V2 (neural network used) - https://arxiv.org/abs/2004.02147
