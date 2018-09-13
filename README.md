# bird-species-classification
Inter species classification

This is an implementation of bird species classification on Python 3 and Keras with Tensorflow backend. The architecture consists of Mask R-CNN and ImageNet models end-to-end. ImageNet models used are Inception V3 and Inception ResNet V2.


**Model trained**
1) Inception V3 features
2) Inception ResNet V2 features


## Data Augmentation
Data Augmentation has been done using [imgaug](https://imgaug.readthedocs.io/en/latest/source/augmenters.html#affine).Table for data Augmentation done for different species is shared in data_augmentation folder.


## Mask R-CNN
Mask R-CNN on the whole image helped to localize birds in the image. Below are the examples of the birds detection from a high resolution image. As the Mask R-CNN is trained on COCO dataset and it has **bird** class, it carves out bird ROIs very perfectly.

![mask_rcnn](https://user-images.githubusercontent.com/22872200/45112827-5b385880-b166-11e8-94c1-8d42edb4a2c6.jpg)


## Drawbacks

Cases where our model failed are depicted in the below images.
![drawbacks](https://user-images.githubusercontent.com/22872200/45113093-0517e500-b167-11e8-9486-f90f8620ae70.jpg)


## Test Results
Results on the test data:

Model Architecture| Data Subset | Train | Validation | Test
------------- | -------- | ---------  | ---------- | ----------
Inception V3  | Images| 91.26 | 13.84|30.95 
Inception V3| Images + Crops| 93.97| 15.50|41.66
Inception ResNet V2  | Images| 97.29 |29.17  |47.96
Inception ResNet V2| Images + Crops |92.29|33.69|49.09

