# -*- coding: utf-8 -*
from matplotlib import pyplot as plt
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad
from keras.callbacks import History, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from keras import backend as k
from keras.wrappers.scikit_learn import KerasClassifier as kclass
import numpy as np
import codecs

#### for hindi text ###
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#####

np.random.seed(1707)

line_length = 248  # seq size


# y_val=y[rand:rand+30]


########## opt params #####################
dropout = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
init = ["normal", "uniform"]
dropout = dict(dropout=dropout)
print dropout
############################# Begin model ###################


def my_model(dropout):
    ############ model params ################
    line_length = 248  # seq size
    train_char = 58
    hidden_neurons = 512  # hidden neurons
    batch = 64  # batch_size
    no_epochs = 5
    ################### Model ################
    model = Sequential()
    # layer 1
    model.add(LSTM(hidden_neurons, return_sequences=True,
                   input_shape=(line_length, train_char)))
    model.add(Dropout(dropout))
    # layer 2
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(Dropout(dropout))
    # layer 3
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(Reshape((248, 512)))
    # fc layer
    model.add(TimeDistributed(Dense(58, activation='softmax')))
    # model.load_weights("weights/model_maha1_noep50_batch64_seq_248.hdf5")
    # model.layers.pop()
    # model.layers.pop()
    # model.add(Dropout(dropout))
    #model.add(TimeDistributed(Dense(train_char, activation='softmax')))
    initlr = 0.00114
    adagrad = Adagrad(lr=initlr, epsilon=1e-08)
    model.compile(optimizer=adagrad,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    ###load weights####
    return model

'''
not working for RNN 
'''
print "Creating model"
classif = kclass(my_model, batch_size=64)
randgrid = RandomizedSearchCV(
    estimator=classif, param_distributions=dropout, n_iter=2)
print "Checking best hyper_params"
print X_train.shape, y_train.shape
opt_result = randgrid.fit(X_train, y_train)
print("Best:{0} using {1}".format(
    opt_result.best_score_, opt_result.best_params_))
