from keras.callbacks import History,ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras.callbacks import Callback
from keras import backend as k
import nump as np


class decay_lr(Callback): 
    '''
    Learning rate decay at end of n epoch

    decay: decay value
    n_epoch: deacy learning rate at end of n_epoch
    '''
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


###keras integrated
def decay_sch(epoch):
    if epoch%6 == 0 :
        lr= 0.10*model.optimizer.lr.get_value()
        return float(lr)
    else:
        return float(model.optimizer.lr.get_value())  


class expdecaylr_loss(Callback):
    '''
    Decay learning rate(lr) exponentially w.r.t loss

    Output: current_lr*e^{loss}
    '''
    def __init__(self):
        super(decaylr_loss, self).__init__()
    def on_epoch_end(self,epoch,logs={}):
        loss=logs.items()[1][1] #get loss
        print "loss: ",loss
        old_lr = self.model.optimizer.lr.get_value() #get old lr
        new_lr= old_lr*np.exp(loss) #lr*exp(loss)
        k.set_value(self.model.optimizer.lr, new_lr)



#decaylr=LearningRateScheduler(decay_sch)


#checkpoint=ModelCheckpoint("weights/adam_noep{0}_batch{1}_seq_{2}.hdf5".format(\
#    no_epochs,batch, seq_length), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')



class lr_printer(Callback):
    '''
    Print lr at beginning of each epoch
    '''
    def __init__(self):
        super(lr_printer, self).__init__()
    def on_epoch_begin(self, epoch, logs={}):
        print('lr:', self.model.optimizer.lr.get_value())


class logger(Callback):
    '''
    Log training metrics in file at end of each epoch
    metrics logged: Loss, Train acc., Val. loss, Val. acc.

    file: filename of logging file
    '''
    def __init__(self,file):
        self.file=file
        super(logger, self).__init__()
    def on_epoch_end(self,epoch,logs={}):
        item=logs.items()
        with open(self.file,"a") as log:
            log.write("------epoch:{0}, lr:{2} ,stats:{1}------\n".format(epoch,item,model.optimizer.lr.get_value()))



### get activations of nth layer
extract_features = theano.function([model.layers[0].input],model.layers[n].output,allow_input_downcast=True)