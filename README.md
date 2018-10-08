# bird-species-classification
Inter species classification

# :relaxed:[***CHALLENGE WINNERS***](http://www.iiitdmj.ac.in/CVIP-2018/BPA.html):relaxed:

This is an implementation of bird species classification challenge hosted by IIT Mandi in [ICCVIP Conference'18](http://www.iiitdmj.ac.in/CVIP-2018/index.html) on Python 3 and Keras with Tensorflow backend. The architecture consists of Mask R-CNN and ImageNet models end-to-end. ImageNet models used are Inception V3 and Inception ResNet V2.

![main_image](https://user-images.githubusercontent.com/22872200/45708132-49ab7380-bb9e-11e8-8bd5-8beb9f077d90.jpg)

The repository includes:
* Generate training data for the model
* Data Augmentation practices
* Generate bird crops using Mask R-CNN 
* Finetuning on Image Net models - Inception V3 & Inception ResNet V2
* Multi- stage Training
* Mask R-CNN and ImageNet Model combined end-to-end for testing purpose
* Evauation file for class averaged precision, recall and F1-scores.

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). 

## Getting Started
 * [modify_data.py](https://github.com/AKASH2907/bird-species-classification/blob/master/modify_data.py) - This code is used to rename the files. <br /> For example: 10 - Type of Data Augmentation 01- Class of Bird 01 - Image Number - 100101.jpg (Image Name)
 * [data_augmentation.py](https://github.com/AKASH2907/bird-species-classification/blob/master/data_augmentation/data_augmentation.py) - Various types of data augmentation used to counter the challenge of large scale variation in illumination,scale, etc. and class imbalance.
 * [create_validation.py](https://github.com/AKASH2907/bird-species-classification/blob/master/create_validation.py) - Used to create Validation data randomly from the augmented training data.
* [gen_train_data_test_data.py](https://github.com/AKASH2907/bird-species-classification/blob/master/gen_train_data_test_data.py) - Generates X_train, Y_train, X_validation, Y_validation, X_test, Y_test
* [inception_v3_finetune.py](https://github.com/AKASH2907/bird-species-classification/blob/master/inception_v3_finetune.py) - Multi-stage Training on Mask R-CNN crops generated and then on data augmented original images.
* [inception_resnet_v2_finetune.py](https://github.com/AKASH2907/bird-species-classification/blob/master/inception_resnet_v2_finetune.py) - Multi-stage Training on Mask R-CNN crops generated and then on data augmented original images resized to 416x416.
* [mask_rcnn/rcnn_crops.py](https://github.com/AKASH2907/bird-species-classification/blob/master/mask_rcnn/rcnn_crops.py) - Localizes bird in the images, crops and then save them for multi-stage learning.
* [mask_rcnn/test_images.py](https://github.com/AKASH2907/bird-species-classification/blob/master/mask_rcnn/test_images.py) - 
End-to-end model of classifying bird specie using Mask R-CNN and **ensembling** of Inception V3 and Inception ResNet V2.
* [evaluate.py](https://github.com/AKASH2907/bird-species-classification/blob/master/evaluate.py) - Calculation of class-averaged precision, recall and F1-scores.

## Step by Step Classification
To help running the model, end to end a docx has been added in case much information about each funcation and thier parameters are required. Here are the steps in summary:

* From the original training data, firstly several data augmentation were taken careof as the dataset contains only 150 images. The number of images were increased to around 1330. The huge number of parameters were unable to learn and generalize in case of validaion data. This also helps in decreasing the effect of class imbalance. Some classes have 6 images whereas some have around 20 images.
* After the data augmentation, validation dataset is created. 10% of each bird species were taken into validation data for model performance testing.
* The model were trained on various Imagenet models such as AlexNet, VGG-16/19, ResNet50, Inception V3 and Inception ResNet V2 with pretrained Imagenet weights. Inception ResNet V2 outperforms them all.
* Multi-stage training comes after that. Used Mask R-CNN to localize birds in the images in their original resolution. Single Shot Detector and YOLO were also used but they needs to be resized images into 416x416 or 512x512 due to which many information is lost. Mask R-CNN code and modules are well explained in this github [repo](https://github.com/matterport/Mask_RCNN). As Mask R-CNN is trained on COCO dataset, and COCO has a class of bird specie, it helped me to crop birds in most of the cases.
* After getting the crops, data augmentation on the cropped images were done and dataset was increased to around 1600 images. Performed multi-stage training with both the dataset of cropped images as well as original images. It helps to improve the accuracy by around 10% in case of Inception V3 model and 2% in Inception ResNet V2.
* Created an architecture end-to-end of Mask R-CNN and Trained Inception models for testing purposes. All the testing images were first passed through Mask R-CNN. After that, it splits into two cases:
  * If the Mask R-CNN is successful to detect bird, which it mostly did, the birdsare cropped from the original and then passed through trained Imagenet models for classification trained on crops and whole images.
  * If the Mask R-CNN fails, then the whole image is classified using pre-trained weights of Imagenet models trained only on original images.
  * It helped to improve the accuracy by 2% from 49 to 51.
  
* After applying Mask R-CNN for both, using confusion matrix Inception V3 performs better in some classes than Inception ResNet V2. Using ensembling, by taking the prediction vector ofboth the models compared them and then finally assign the class to the image whosoever has the highest prediction for certain species. This helped to improve the accuracy by almost 5% from 51 to around 56%. Tables are dicussed below.

Some important sub-parts are discussed below:
## Dataset
In this repo, I have used the dataset fom the [ICCVIP'18](http://www.iiitdmj.ac.in/CVIP-2018/challenges.html) Bird Species Classification Challenge. Training dataset contains 150 images and test dataset contains 158 images with 1 image corrupted. There are total 16 species of birds to be classified. The resolution of the images lies in between 800x600 to 6000x4000.
## Data Augmentation
Data Augmentation has been done using [imgaug](https://imgaug.readthedocs.io/en/latest/source/augmenters.html#affine).Table for data Augmentation done for different species is shared in [data_augmentation folder](https://github.com/AKASH2907/bird-species-classification/tree/master/data_augmentation).

## Mask R-CNN
Mask R-CNN on the whole image helped to localize birds in the image. Below are the examples of the birds detection from a high resolution image. As the Mask R-CNN is trained on COCO dataset and it has **bird** class, it carves out bird ROIs very perfectly. More than 140 images were able to give successfull cropped bird images out of 150 images.

![mask_rcnn](https://user-images.githubusercontent.com/22872200/45112827-5b385880-b166-11e8-94c1-8d42edb4a2c6.jpg)


## Challenges

As a new dataset always have some problems whereas some major challenges too: 
1) The training dataset mostly contains bird images in which bird were almost 10-20% of the whole image whereas in case of test images the bird contains 70-80% of the image. Sometimes, the model fails to detect the birds due to less number of birds in the dataset.
2) In some classes the birds cover not even 10% of the whole images or the colour of bird and surrounding are ver similar. Cases where birds are brown in colour. In those cases, model fails to localize birds due to occlusion problem or background similarity problems. Some cases as follows:  
![drawbacks](https://user-images.githubusercontent.com/22872200/45113093-0517e500-b167-11e8-9486-f90f8620ae70.jpg)

## Experiments 

I tried multi-stage training with training on original images first and then on crops and on crops and then on original images. Firstly, training on the images and then on the crops gave the better results. As well as for testing, we use Inception V3 crops weights and Inception ResNet V2 crops+images weights to identify the specie of bird. <br />
Please find the weight file for 7 epochs a follows:<br />
[1] [inception_v3_crops.h5](https://drive.google.com/open?id=1aZbAKPoKTlZ3vVjhB120veu_YoSxQl6X) - Trained only on cropped images. <br />
[2] [inception_v3_crops+images.h5](https://drive.google.com/open?id=1kv76lIq4BQ0NycATrrMrLAD9SxnIR_AN) - Trained on Images plus crops. <br />
[3] [inception_resnet_v2_images.h5](https://drive.google.com/open?id=1ABTrngtdJLBEwFYbRh4O6CRbTWHh-X6z) - Trained on Images only. <br />
[4] [inception_resnet_v2_images+crops.h5](https://drive.google.com/open?id=1s6gXZ_i92qoXalx44NPgC30VMpsOyjih) - Trained on Images + crops for 7 epochs. <br />

We could have trained it for more epochs but it was not giving significant iprovements in the results at all. 

## Model Architecture

The architecture of the model is as below:
![model_architecture](https://user-images.githubusercontent.com/22872200/46626495-29dfed80-cb55-11e8-97e1-284bafcec51c.png)


## Test Results
Results on the test data after Multi-stage training:

Model Architecture| Data Subset | Train | Validation | Test
------------- | -------- | ---------  | ---------- | ----------
Inception V3  | Images| 91.26 | 12.76|30.95 
Inception V3| Images + Crops| 93.97| 15.50|41.66
Inception ResNet V2  | Images| 97.29 |29.17  |47.96
Inception ResNet V2| Images + Crops |92.29|33.69|49.09

Evaluation on test data in terms of class-averaged Precision, Recall and F1-scores:

Model Architecture| Precision | Recall | F1
------------- | -------- | ---------  | ---------- 
Mask R-CNN + Inception V3  |  48.61 | 45.65|47.09 
Mask R-CNN + Inception ResNet V2|  53.62| 48.72|51.05
Mask R-CNN + Ensemble   |  **56.58** |**54.8**  |**55.67**

Final Confusion Matrix:
![final_confusion_matrix](https://user-images.githubusercontent.com/22872200/45716831-b4b47480-bbb5-11e8-9d76-e576dfb8cc11.jpeg)

Hope it helps!!! If youmake any progress on the dataset or face any problems, please let me know. :relaxed:

The dataset is uploaded on Kaggle and the link is shared as follow:
[Dataset](https://www.kaggle.com/akash2907/bird-species-classification) <br />

Description of all the codes have been shared in this [PDF](https://github.com/AKASH2907/bird-species-classification/blob/master/Codes_info.pdf)
## References
[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, "[
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" arXiv preprint arXiv:1512.00567. <br />
[2] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, "[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)" arXiv preprint arXiv:1602.07261. <br />
[3] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, "[Mask R-CNN](https://arxiv.org/abs/1703.06870)" arXiv preprint arXiv:1703.06870 
