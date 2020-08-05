# Prepare-Image-Datasets-for-CNN
Prepare-Image-Datasets-for-CNN

***********************************************************************************************************************
This is a python code to read the images from a given directory and prepare training set and test set suitable for CNN
***********************************************************************************************************************

Package Version\
python 3.6.8\
pandas 0.24.1\
Keras 2.2.4\
skimage 0.16.2

This python code is most useful to read the content of a directory\
and prepare images for train and test set.\
The train set and test are input to Keras CNN deep learning models.

The input to this code is the path to a directory containing images.\
Output of this code is train and test sets along with class labels.

### How to run this python code?
Please provide the path to the director containing images to imFolder.\
As an example, a few images are kept in the folder 'images'.\
Few samples are kept here for complete dataset please refer\
https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/\
Also it is required to change to current directory using os.chdir().\
The ground truth i.e class label information is to be kept in the file groundTruthMv_a.txt.

Initially the class label information is read and stored in a dataframe.\
All the paths of the images from the given directory are read and a list of paths to images is prepared into allImsPaths.\
This list is then refined to remove the paths which are not part of ground truth file.

A directory is created by storing images content and class label of the image. The prepared directory is called as imDbDict.\
The images samples are then split into training and testing samples.\
Then the training and testing sets of images are prepared using prepareTrainAndTestData, which are suitable for CNN.\
This will prepare train images set and test images set as numpy array .\
This n dimensional numpy array has dimension of (num of Samples, dimensions of the image).\
The target class labels are prepared as y_trainSet and y_testSet with dimension of (number of samples, number of classes).

# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passioinate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com
