# OCR (Optical Character Recognition) using MNIST Dataset

This repository contains code and resources for building an Optical Character Recognition (OCR) system using the MNIST dataset. The MNIST dataset is a widely used dataset of handwritten digits, which makes it suitable for developing and testing OCR algorithms.

Here, I have added the python notebook of implementation of model trained basis of the MNIST handwritten charcters. Also, the documentation regarding the exploration I had done in this domain.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Optical Character Recognition (OCR) is the process of converting images of handwritten or printed text into machine-encoded text. This repository demonstrates how to build a basic OCR system using the MNIST dataset. The idea is to treat each handwritten digit as a character and develop a simple recognition pipeline.

## Prerequisites

Before you begin, ensure you have the following:

- Python (>=3.6)
- Virtual environment (optional but recommended)

## Results

The OCR system will display the recognized digits along with their predicted labels from the MNIST dataset.

Example input:
```
input_prediction = model.predict(final_img)
print(input_prediction)
plt.imshow(resized_img)
print('The digit predicted by the model is: ', np.argmax(input_prediction))
```

Example output:
```
The digit predicted by the model is:  3
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create a pull request.

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push the changes to your fork.
5. Create a pull request describing your changes.
