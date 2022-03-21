# tfmars

<p align="center">
    <img src="tfmars-logo.PNG" width=128>
</p>

**tfmars** is the TensorFlow's implementation of Mobile-aware Convolutional Neural Network for Sensor-based Human Activity Recognition, a sibling of [tfgarden](https://github.com/Shakshi3104/tfgarden).
In this repository, some CNN models implemented in tfgarden have been implemented as Attention insertable models. 
Also, **MarNASNets** has been implemented.

MARS means **M**obile-aware **A**ctivity **R**ecognition model**S**.

## Models

- Simple CNN: used on [the paper by Li et al](https://www.mdpi.com/1424-8220/18/2/679).
- VGG16
- Inception v3
- ResNet 18
- PyramidNet 18
- Xception
- DenseNet 121
- MobileNet
- MobileNetV2
- MobileNetV3 Small
- NASNet Mobile
- MnasNet
- EfficientNet B0
- EfficientNet lite0

### MarNASNets

**MarNASNets** are the CNN architectures designed by using Bayesian-optimization Neural Architecture Search via Keras Tuner.
MarNASNets are **mobile-aware** models that achieves higher accuracy with fewer parameters than existing models.
There are variations with different search spaces (A - E).

## Install

```bash
pip install git+https://github.com/Shakshi3104/tfmars.git
```

## Dependency

- `tensorflow >= 2.4.1`

## Citation 

Under construction...
