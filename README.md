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
Data Augmentation has been done using [imgaug](https://imgaug.readthedocs.io/en/latest/source/augmenters.html#affine).Table for data Augmentation of bird species:

|Species| Gaussian Noise | Gaussian Blur|Flip|Contrast|Hue|Add|Multiply|Sharp|Affine|Total|
|--------------------|----------------| -------------|----|--------|---|---|--------|-----|------|-----|
|blasti| ✔|✔|✔|✔|✔|✘|✘|✘|✘|90|
|bonegl| ✔|✔|✔|✔|✔|✔|✔|✔|✔|78|
brhkyt | ✔|✔|✔|✔|✔|✔|✔|✔|✔|65|
cbrtsh| ✔|✔|✔|✔|✔|✔|✔|✔|✔|91|
cmnmyn| ✔|✔|✔|✔|✔|✔|✔|✔|✔|91|
gretit| ✔|✔|✔|✔|✔|✔|✔|✔|✔|78|
hilpig| ✔|✔|✔|✔|✔|✔|✔|✘|✘|80|
himbul| ✔|✔|✔|✔|✔|✘|✘|✘|✘|99|
himgri| ✔|✔|✔|✔|✔|✘|✘|✘|✘|100|
hsparo| ✔|✔|✔|✔|✔|✘|✘|✘|✘|81|
indvul| ✔|✔|✔|✔|✔|✘|✘|✘|✘|81|
jglowl| ✔|✔|✔|✔|✔|✔|✔|✔|✔|78|
lbicrw| ✔|✔|✔|✔|✔|✔|✔|✔|✔|78|
mgprob| ✔|✔|✔|✔|✔|✔|✔|✔|✔|78|
rebimg| ✔|✔|✔|✔|✔|✔|✔|✘|✘|80|
wcrsrt| ✔|✔|✔|✔|✔|✔|✔|✘|✘|80|
