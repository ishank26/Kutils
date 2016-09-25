from keras.callbacks import History,ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras.callbacks import Callback
from keras import backend as k

#lr_sch

###self
class decay_lr(Callback): 
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch=n_epoch
        self.decay=decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.get_value()
        if epoch > 1 and epoch%self.n_epoch == 0 :
            new_lr= self.decay*old_lr
            k.set_value(self.model.optimizer.lr, new_lr)
        else:
            k.set_value(self.model.optimizer.lr, old_lr)
###keras
def decay_sch(epoch):
    if epoch%6 == 0 :
        lr= 0.10*model.optimizer.lr.get_value()
        return float(lr)
    else:
        return float(model.optimizer.lr.get_value())  

decaylr=LearningRateScheduler(decay_sch)


#checkpoint            
checkpoint=ModelCheckpoint("weights/adam_noep{0}_batch{1}_seq_{2}.hdf5".format(\
    no_epochs,batch, seq_length), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')


#lr printer
class lr_printer(Callback):
    def __init__(self):
        super(lr_printer, self).__init__()
    def on_epoch_begin(self, epoch, logs={}):
        print('lr:', self.model.optimizer.lr.get_value())