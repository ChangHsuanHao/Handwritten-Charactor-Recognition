# Comparison of Machine Learning Algorithms and Handwritten Character Recognition (OCR)
## Abstract
The topic compares the accuracy and training time of five common machine learning algorithms, and selects the algorithm with the highest accuracy rate, combined with the Canny edge detection operator, to implement multi-object tracking for handwritten character recognition, which can recognize both upper and lower case letters and numbers in English.
## Paper
[Comparison of Machine Learning Algorithms and Handwritten Character Recognition (OCR)](https://1drv.ms/b/c/0f31cf811556b421/EcOjX0TK2zBIq8Jo0f1HSU4B-3Kj5I8b0cfncuhnVD6YCg?e=ueXSQz)
## Database
tensorflow EMNIST and MNIST
## Comparison of Machine Learning Algorithms
### Data
* Image size: 28×28 pixel
* Training sets: 60,000 images
* Testing sets: 10,000 images
### Algorithms
* CNN 
* SVM (Support vector machine)
* RF (Random Forest) 
* KNN 
* DT (Decision Tree)
### Result
* Accuracy: CNN > SVM > RF > KNN > DT
* Training Time: KNN << DT < RF << CNN < SVM
* CNN is the best👍


|     Algo.     |  CNN   |  KNN   |  SVM   |   DT   |   RF   |
|:-------------:|:------:|:------:|:------:|:------:|:------:|
|   Accuracy    | 0.9876 | 0.9688 | 0.9792 | 0.8786 | 0.9709 |
| Time (second) | 166.6  | 0.097  | 295.4  |  21.9  |  42.4  |

## Handwritten Character Recognition (OCR)

### Goal
Recognizing several handwritten characters on a paper through a laptop camera by CNN model.

### Model Building (cnn_model_builder_byclass & cnn_model_builder_bymerge)
#### Data Description: EMNIST
* Byclass 
    * 697932 training images
    * 116323 testing images
    * 62 unbalanced classes
* Bymerge 
    * 697932 training images
    * 116323 testing images
    * 62 unbalanced classes
* Image sizes are all 28 * 28 pixel
#### Data Preprocessing
#### Model: CNN
* Structure
<img src="https://hackmd.io/_uploads/Hkq2s_4oA.png" width="50%">

* Comparison

|  Data   | Test Accuracy | Test Lost |
|:-------:|:-------------:|:---------:|
| Byclass |    0.8630     |  0.3763   |
| Bymerge |    0.8983     |  0.2817   |

→ Chose **Bymerge** as training data

### Image Processing and Prediction (object_track_cam)
* Image Collection
    * Use OpenCV to turn on the camera.
    * Set up trackers and start tracking.
    * Enclose the tracked characters in boxes to highlight them.
* Image Processing : Canny edge detector
    * Gaussian Blur
    * Gradient Computation
    * Non-maximum Suppression
    * Double Thresholding 
* Prediction
    * Using the trained CNN model.
    * Output the results above the tracked character.
* Result
[Demo video](https://1drv.ms/v/c/0f31cf811556b421/EVpOe69pRstOrbu5fQaQAMYB_YUrncUIb0llOhpVANH2QA?e=VjCY8Z)
    * The ambient light source has a significant impact on the prediction results. To achieve good results, it is essential to have sufficient lighting to illuminate the paper and ensure that the background is dark enough.
    * The paper should be kept as flat as possible and face the camera directly to avoid shadows that could affect image recognition.

![image](https://hackmd.io/_uploads/SktKNYVsC.png)
