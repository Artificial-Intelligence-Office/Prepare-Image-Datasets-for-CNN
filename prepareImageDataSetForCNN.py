import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import pandas as pd
from keras.utils import to_categorical

#change to current directory
os.chdir('')
#folder containing images
imFolder='images'
#ground truth file
grndTruthPath='groundTruthMv_a.txt'

#change dimension of the image
dim=(256,256)
input_shape = (dim[0],dim[1],3)
num_classes = 2

#collect all the images present in the folder
dirContent=os.listdir(imFolder)
#column name to be read from ground truth file
clNames=['Image','Sentiment']
clsLabels=pd.read_csv(grndTruthPath,names=clNames,delimiter='\t')
clsLabels.set_index('Image',inplace=True)

#Read the file name from the given image
#check the corresponding class present in the ground truth file by looking into dataframe clsLabels
#in this example there are two classes positive and negative, the positive class is set to 1 and negative is set to 0
#read the images from the folder perform preprocessing such as dimension change, gray2rgb
#store the both image and class into the dictonary imDbDict
def createImagesSet(allImagesFoldrPath,dim,clsLabels):
    x_imageSet=np.empty((len(allImagesFoldrPath),dim[0],dim[1],3))
    imDbDict={}
    y_Set=np.empty((len(allImagesFoldrPath),1))
    for im in range(len(allImagesFoldrPath)):
        readImage=imread(allImagesFoldrPath[im])
        print(allImagesFoldrPath[im])
        imNamge=allImagesFoldrPath[im].split('/')[-1].split('.jpg')[-2]
        actualClass=clsLabels.loc[imNamge,'Sentiment']
        
        if ('positive' in actualClass):
            print('\t\tpositive class')
            y_Set[im]=1
        else:
            print('\tnegative class')
            y_Set[im]=0
            
        if (len(readImage.shape)>=3):
            if readImage.shape[2]>3:
                readImage=readImage[:,:,:3]            
        else:
            print(im,readImage.shape)
            readImage=gray2rgb(readImage)            
        readImage=resize(readImage,dim)
        x_imageSet[im]=readImage
        imDbDict[allImagesFoldrPath[im]]=(x_imageSet[im],y_Set[im])
    return imDbDict

#Check whether the given image is part of the ground truth file
#collect all the names of the images into two category images present in ground truth and another not present in ground truth
def collectImNames(entireDb):
    imNmPresentInGrndTrth=[]
    imPathNotPresentInGrndTrth=[]
    for imPath in range(len(entireDb)):
        imNm=entireDb[imPath].split('/')[-1].split('.jpg')[-2]
        print(imNm)
        if imNm in clsLabels.index:
            imNmPresentInGrndTrth.append(imNm)
        else:
            imPathNotPresentInGrndTrth.append(entireDb[imPath])
    return imNmPresentInGrndTrth,imPathNotPresentInGrndTrth

#Prepare the train and test set based on the path for training samples and test samples are received.
#It utilizes keras tool to_categorical to convert numerical class into one hot representation
def prepareTrainAndTestData(allImagesTrainPath,allImagesTestPath,imDbDict):
    x_trainImSet=np.empty((len(allImagesTrainPath),dim[0],dim[1],3))
    x_testImSet=np.empty((len(allImagesTestPath),dim[0],dim[1],3))
    y_trainSet=np.zeros(len(allImagesTrainPath))
    y_testSet=np.zeros(len(allImagesTestPath))
    for trnPi in range(len(allImagesTrainPath)):
        (x_trainImSet[trnPi],y_trainSet[trnPi])=imDbDict[allImagesTrainPath[trnPi]]
    
    for testPi in range(len(allImagesTestPath)):
        (x_testImSet[testPi],y_testSet[testPi])=imDbDict[allImagesTestPath[testPi]]
        
    x_trainImSet= x_trainImSet.astype('float32')
    x_testImSet= x_testImSet.astype('float32')
    x_trainImSet /= 255.0
    x_testImSet /= 255.0

# convert class vectors to matrices as binary
    y_trainSet= to_categorical(y_trainSet, num_classes)
    y_testSet= to_categorical(y_testSet, num_classes)
    
    print('Number of samples in training set ', x_trainImSet.shape[0])
    print('Number of samples in test set', y_testSet.shape[0])
    
    return (x_trainImSet,y_trainSet), (x_testImSet,y_testSet)

#prepare a list containig all image files
allImsPaths=[(imFolder+'/'+di) for di in dirContent if('txt' not in di)]
imNmPresentInGrndTrth,imPathNotPresentInGrndTrth=collectImNames(allImsPaths)
labels=list(clsLabels.loc[imNmPresentInGrndTrth,'Sentiment'])
#remove images which are not part of ground truth
for rPath in imPathNotPresentInGrndTrth:
    allImsPaths.remove(rPath)

#create a dictonary containing images and class
imDbDict=createImagesSet(allImsPaths,dim,clsLabels)

#divide 50% of images into training set and another 50% into test set.
trainSetImagesPath=allImsPaths[:int(len(allImsPaths)/2)]
testSetImagesPath=allImsPaths[int(len(allImsPaths)/2):]
#Prepare train images set and test images set as numpy array 
#This n dimensional numpy array has dimension of (num of Samples, dimensions of the image)
#The target class labels are prepared as y_trainSet and y_testSet with dimension of (number of samples, number of classes)
(x_trainImSet,y_trainSet), (x_testImSet,y_testSet)=prepareTrainAndTestData(trainSetImagesPath,testSetImagesPath,imDbDict)
