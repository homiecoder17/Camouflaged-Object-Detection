# Camouflaged Object Detection Using CNN

## Overview

This project detects whether an image contains a camouflaged object using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Objective

To classify images into:

- `Camouflage Detected`
- `No Camouflage`

## Method

- Load image dataset using TensorFlow
- Resize images to `128 x 128`
- Normalize pixel values to the range `0-1`
- Train a CNN model using `Conv2D` and `MaxPooling2D` layers
- Predict the result on a new image

## Dataset Structure

```text
dataset/
  Train/
    camouflage/
    normal/
```

## Requirements

- Python
- TensorFlow
- NumPy

## Run

```bash
python dl.py
```

## Output

- Model trains on the dataset
- Predicts one of the following:
  - `Camouflage Detected`
  - `No Camouflage`

## Status

Project status: `60%` complete.

## Future Work

- Improve model accuracy
- Add validation and improve dataset
- Save trained model
- Add graphs for accuracy and loss
