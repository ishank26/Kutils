import random
import numpy as np

np.random.seed(1707)

'''
with open("train.txt") as t:
        files = t.readlines()
        img = []
        label = []
        for lines in files:
            item = lines.split()
            img.append(item[0])
            label.append(int(item[1]))
        pos=[]
        pos= [i for i in range(len(img))]
        np.random.shuffle(pos)
        label=[label[i] for i in pos]
        img=[img[i] for i in pos]
        print label 
        print img       
'''

def shuffleData(data, labels):
    pos= [i for i in range(len(labels))]
    np.random.shuffle(pos)
    data=[data[i] for i in pos]
    labels=[labels[i] for i in pos]
    return data, labels

