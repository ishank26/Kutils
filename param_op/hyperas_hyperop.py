# -*- coding: utf-8 -*
from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad
from keras.callbacks import History,LearningRateScheduler,ModelCheckpoint,EarlyStopping
from keras.callbacks import Callback
from keras import backend as k
import numpy as np
import codecs

#### for hindi text ###
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

np.random.seed(1707)



#y_val=y[rand:rand+30]

########## opt params #####################
dropout=np.array([0.4,0.5,0.6,0.7,0.8])
init=["normal", "uniform"]
dropout=dict(dropout=dropout)
print dropout
############################# Begin model ###################
def my_model(X_train,y_train,X_test,y_test):
    ############ model params ################
    line_length = 248 # seq size
    train_char = 58
    hidden_neurons = 512 # hidden neurons
    batch = 64  #batch_size
    no_epochs= 3
    ################### Model ################

    ######### begin model ########
    model = Sequential()
    # layer 1
    model.add(LSTM(hidden_neurons, return_sequences=True,input_shape=(line_length, train_char)))
    model.add(Dropout({{choice([0.4,0.5,0.6,0.7,0.8])}}))
    # layer 2
    model.add(LSTM(hidden_neurons,return_sequences=True))
    model.add(Dropout({{choice([0.4,0.5,0.6,0.7,0.8])}}))
    #layer 3
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(Dropout({{choice([0.4,0.5,0.6,0.7,0.8])}}))
    #fc layer
    model.add(TimeDistributed(Dense(train_char, activation='softmax')))
    model.load_weights("weights/model_maha1_noep50_batch64_seq_248.hdf5")
    ########################################################################
    checkpoint=ModelCheckpoint("weights/hypmodel2_maha1_noep{0}_batch{1}_seq_{2}.hdf5".format(no_epochs, batch, line_length), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
    
    initlr= 0.00114
    adagrad=Adagrad(lr=initlr, epsilon=1e-08,clipvalue={{choice([0,1,2,3,4,5,6,7])}})
    model.compile(optimizer=adagrad, loss='categorical_crossentropy', metrics=['accuracy'])
    history = History()
    #fit model
    model.fit(X_train, y_train, batch_size=batch, nb_epoch=no_epochs, validation_split=0.2,callbacks = [history, checkpoint])
    
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

print "Getting best hyper_params"
best_run, best_model = optim.minimize(model=my_model,data=get_data,algo=rand.suggest,max_evals=20,trials=Trials())
print(best_run,"\n")
print(best_model)