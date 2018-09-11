# Birds localization

ROI of birds in the images.

## SSD (Single Shot Detector)

## Saliency Detection

## Mask R-CNN
Mask R-CNN is adding a branch to Faster R-CNN that outputs a binary mask that says whether or not a given pixel is part of an object. The branch (in white in the above image), as before, is just a Fully Convolutional Network on top of a CNN based feature map. 
Mask R-CNN is trained on COCO dataset.COCO datset has a class bird which helped us to crop bird crops from the original images.
