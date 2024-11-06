# COVID-19 Mask Detection Project

This repository contains code and documentation for a COVID-19 mask detection project. The project utilizes machine learning models to classify images of individuals wearing masks correctly, improperly, or not wearing a mask at all. 

## Folder Structure

- **Code/**
  - `Code.ipynb`: Main notebook containing shared functions and utilities used across models.
  - `cnn.ipynb`: Implementation of a Convolutional Neural Network (CNN) for mask detection.
  - `hog-kNN.ipynb`: K-Nearest Neighbors (KNN) model with Histogram of Oriented Gradients (HOG) features.
  - `hog-mlp.ipynb`: Multilayer Perceptron (MLP) model using HOG features.
  - `hog-svm.ipynb`: Support Vector Machine (SVM) model with HOG features.
  - `test_functions.ipynb`: Contains additional functions for testing model performance.

- **Models/**
  - `Computer Vision Report.pdf`: Detailed report of the project, including dataset details, methodology, results, and discussions.
  - `README.md`: Documentation for the models and project overview.

## Dataset

The dataset used for this project is a COVID-19 mask detection dataset, containing images labeled as "mask," "no_mask," and "improper_mask." The dataset is divided into training and testing sets, with images resized to 224x224x3 RGB format. Preprocessing steps include resizing and label encoding.

For more details on the dataset, refer to the report or access the dataset through the [Google Drive link](https://drive.google.com/drive/folders/1zRYOcP13nh3Aw-smSF3n7LgfVS6LepaK?usp=share_link).

## Implemented Models

1. **Support Vector Machine (SVM)**: Utilizes HOG features to classify images. Optimized with an RBF kernel.
2. **Multilayer Perceptron (MLP)**: HOG features with a neural network structure, optimized using GridSearchCV.
3. **K-Nearest Neighbors (KNN)**: Simple, non-parametric model trained on HOG features, optimized for the best `k` value.
4. **Convolutional Neural Network (CNN)**: End-to-end deep learning model directly trained on images for higher accuracy.

Each model is evaluated on performance metrics including accuracy, F1 score, precision, and recall.

## Results

The CNN model achieved the highest accuracy (92.79%), demonstrating superior performance in detecting mask usage compared to the HOG-based models. The detailed results, including performance metrics and discussion on model biases, can be found in the report under `Models/Computer Vision Report.pdf`.

| Model       | Accuracy | F1 Score | Precision | Recall |
|-------------|----------|----------|-----------|--------|
| HOG + SVM   | 84.72%   | 78.00%   | 72.00%    | 85.00% |
| HOG + MLP   | 86.24%   | 86.05%   | 86.66%    | 86.24% |
| HOG + KNN   | 87.77%   | 86.00%   | 87.00%    | 88.00% |
| CNN         | 92.79%   | 92.69%   | 92.89%    | 92.79% |

## Additional Notes

- **Limitations**: Only some files may be available due to file size restrictions. For access to all resources, refer to the Google Drive link above.
- **References**: For model methodologies and other technical details, see references in the report.

## Contact

For questions or further details, please reach out to the author.
