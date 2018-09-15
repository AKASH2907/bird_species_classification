# bird-species-classification
Inter species classification

This is an implementation of bird species classification on Python 3 and Keras with Tensorflow backend. The architecture consists of Mask R-CNN and ImageNet models end-to-end. ImageNet models used are Inception V3 and Inception ResNet V2.

The repository includes:
* Generate training data for the model
* Data Augmentation practices
* Generate bird crops using Mask R-CNN 
* Finetuning on Image Net models - Inception V3 & Inception ResNet V2
* Muli- stage Training
* Mask R-CNN and ImageNet Model combined end-to-end for testing purpose
* Evauation file for class averaged precision, recall and F1-scores.

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). 

## Getting Started
 * [modify_data.py](https://github.com/AKASH2907/bird-species-classification/blob/master/modify_data.py) - This code is used to rename the files. For example: 10 - Type of Data Augmentation 01- Class of Bird 01 - Image Number - 100101.jpg (Image Name)
 * [data_augmentation.py](https://github.com/AKASH2907/bird-species-classification/blob/master/data_augmentation/data_augmentation.py) - Various types of data augmentation used to counter the challenge of large scale variation in illumination,scale, etc. and class imbalance.
 * [create_validation.py](https://github.com/AKASH2907/bird-species-classification/blob/master/create_validation.py) - Used to create Validation data randomly from the augmented training data.
* [gen_train_data_test_data.py](https://github.com/AKASH2907/bird-species-classification/blob/master/gen_train_data_test_data.py) - Generates X_train, Y_train, X_validation, Y_validation, X_test, Y_test
* [inception_v3_finetune.py](https://github.com/AKASH2907/bird-species-classification/blob/master/inception_v3_finetune.py) - Multi-stage Training on Mask R-CNN crops generated and then on data augmented original images.
* [inception_resnet_v2_finetune.py](https://github.com/AKASH2907/bird-species-classification/blob/master/inception_resnet_v2_finetune.py) - Multi-stage Training on Mask R-CNN crops generated and then on data augmented original images resized to 416x416.

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

## References
[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, "[
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" arXiv preprint arXiv:1512.00567. <br />
[2] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, "[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)" arXiv preprint arXiv:1602.07261. <br />
[3] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, "[Mask R-CNN](https://arxiv.org/abs/1703.06870)" arXiv preprint arXiv:1703.06870 
