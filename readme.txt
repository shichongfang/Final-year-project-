Human Protein Atlas Image Classification
========================================


Introduction
------------

In this task, It will develop model which is capable of classifying mixed patterns of proteins in microscope images.
The model will be used by The Human Protein Atlas to build a tool integrated with their smart-microscopy system to
identify a protein's location(s) from a high-throughput image.


Environment
-----------

* Linux with GPU (cuda 8.0)
* Anaconda
* Python 2.7


Setup
-----

1. Install required python libraries:

    ```
    pip install numpy opencv-python scikit-learn mxnet-cu90==1.0.0
    ```

2. Download data from kaggle:

    link: https://www.kaggle.com/c/human-protein-atlas-image-classification/data

3. Download pre-trained resnet-101 model on imagenet:

    link: http://data.dmlc.ml/mxnet/models/imagenet/resnet/101-layers/

4. Modify 0_path.py for data and model path.


Usage
-----

Execute each step in order:

    ```
    # check configuration path
    python 0_path.py
    
    # image preprocessing
    python 1_image_preprocessing.py
    
    # feature extraction
    python 2_feature_extraction.py
    
    # train image classifier
    python 3_train.py
    
    # predict test image
    python 4_predict.py
    ```


Method
------

1. Image Preprocessing

    Download Human Protein Atlas Image Dataset from Kaggle, read images using Python OpenCV.
    Visualize some huamn protein atlas images to have a view of the whole dataset and task.
    Split images in the dataset to train, and test sets, resize images and do some preprocessing.
    Label images in train set with labels.

2. Feature Extraction

    Extraction huamn protein atlas image features. There are many kinds of features to use.
    Using deep neural network features in this task, for example, ResNet 101 pre-trained
    on imagenet dataset.

3. Train Image Classifier

    Train human protein atlas image classifier to classify images. There are many types of classifiers.
    Using traditional classifier Support Vector Machine (SVM) in this task, with deep neural network features as input.

4. Predict Test Image

    Having trained image classifier,  Using Support Vector Machine for prediction.
    Input test images into the classifier and get the output labels, which are classification results for human protein atlas images.

