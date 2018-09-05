# bird-species-classification
Inter species classification

**Model trained**
1) Alexnet features
2) VGG16/19 features

**Model in training**
1) Siamese Network
2) Triplet Network

**Ideas to Review**
1) ResNet-50 
2) Global Context vs Local context (Strip wise)

Code for reviewing:- 
Have a look at the make_pairs.py. A problem in the batch tensor generation.

## Data Augmentation
Data Augmentation has been done using [imgaug](https://imgaug.readthedocs.io/en/latest/source/augmenters.html#affine).Table for data Augmentation done for different species is shared in data_augmentation folder.


## Mask R-CNN
Mask R-CNN on the whole image helped to localize birds in the image. Below are the examples of the birds detection from a high resolution image. As the Mask R-CNN is trained on COCO dataset and it has **bird** class, it carves out bird ROIs very perfectly.

![mask_rcnn](https://user-images.githubusercontent.com/22872200/45112827-5b385880-b166-11e8-94c1-8d42edb4a2c6.jpg)


## Drawbacks

Cases where our model failed are depicted in the below images.
![drawbacks](https://user-images.githubusercontent.com/22872200/45113093-0517e500-b167-11e8-9486-f90f8620ae70.jpg)
