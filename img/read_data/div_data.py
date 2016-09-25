import cv2, random
import numpy as np
import glob

np.random.seed(1707)


def shuffleData(data, labels):
    pos= [i for i in range(len(labels))]
    np.random.shuffle(pos)
    data=[data[i] for i in pos]
    labels=[labels[i] for i in pos]
    return data, labels

def loadData(datafile):
    with open(datafile) as f:
        files = f.readlines()
        img = []
        label = []
        for lines in files:
            item = lines.split()
            img.append(item[0])
            label.append(int(item[1]))
        return img, label

def loadImages(path_list, crop_size=224, shift=15):
    images = np.ndarray([len(path_list),3, crop_size, crop_size])
    for i in xrange(len(path_list)):
        img = cv2.imread(path_list[i])
        h, w, c = img.shape
        assert c==3
        img = img.astype(np.float32) 
        img= img.transpose((2,0,1))
        images[i] = img
    return images


image_list, label_list= loadData("/home/data/224_align_col/labels.txt")

image_list, label_list= shuffleData(image_list,label_list)


print len(image_list)
#train
train_path=image_list[:2241]
train_label=label_list[:2241]
#val
val_path=image_list[2241:2721]
val_label=label_list[2241:2721]
#test
test_path=image_list[2721:]
test_label=label_list[2721:]

#load train images
X_train=loadImages(train_path)
#y_train=np_utils.to_categorical(train_label, nb_classes)

#load test images
X_test=loadImages(test_path)
#y_test=np_utils.to_categorical(test_label, nb_classes)
#load val images
X_val=loadImages(val_path)
#y_test=np_utils.to_categorical(val_label, nb_classes)