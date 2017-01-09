Kutils
==========
Utility functions for Keras.  


### [Callbacks](https://github.com/ishank26/Kutils/blob/master/callbacks/helper.py) 
* **Exp_decay learning rate:** Exponential decay of learning rate w.r.t loss, after n epochs.
* **Decay learning rate:** Anneal learning at end of nth epoch by custom value.
* **Learning rate printer.**
* **Training metrics logger.**
* **Get activations of nth layer.**  

### Image                                                      
* **Preprocessing**
    * [Random rotate, Random shift (Data augmentation)](https://github.com/ishank26/Kutils/blob/master/img/preprocess/data_augment.py)
    * [Affine transform](https://github.com/ishank26/Kutils/blob/master/img/preprocess/data_augment.py)
    * [Face alignment](https://github.com/ishank26/Kutils/blob/master/img/preprocess/face_align2.py)
* **Read data**
    * [Output data](https://github.com/ishank26/Kutils/blob/master/img/read_data/div_data.py)
    * [Label to txt](https://github.com/ishank26/Kutils/blob/master/img/preprocess/label.py)
    * [Shuffle data](https://github.com/ishank26/Kutils/blob/master/img/read_data/div_data.py)  
    
### NLP
* [**everything2vec:**](https://github.com/ishank26/Kutils/blob/master/nlp/every2vec.py) A library to integrate word2vec and data processing functions.  

### Models
* [**feature-SVM:**](https://github.com/ishank26/Kutils/blob/master/models/activ_cnn.py) Apply SVM to nth layer activations.

### Hyper_params Optimization
* [**skopt:**](https://github.com/ishank26/Kutils/blob/master/param_op/skopt.py) Apply GridSearch, RandomSearch using sklearn. (Not working for RNN)
* [**hypopt:**](https://github.com/ishank26/Kutils/blob/master/param_op/hypopt.py) Apply GridSearch, RandomSearch using hyperas library.

&nbsp;


**Note:** Repo under development. Mail me for any info or contribute :)
