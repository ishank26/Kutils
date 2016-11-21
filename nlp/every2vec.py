# -*- coding: utf-8 -*
import re
import codecs
import json
import numpy as np
import sys
from gensim.models import Word2Vec
from keras.initializations import glorot_uniform


### if utf-8 charcaters, reload system (failsafe)
reload(sys) 
sys.setdefaultencoding('utf8')
###

class every2vec(object):

    def __init__(self,model_path):
        self.model_path = model_path
        self.model = Word2Vec.load(self.model_path)
        self.model.init_sims(replace=True)

    ### build corpus  
    def built_corp(self, filepath):
        '''
        Build corpus from filepath
        '''
    	corpus=[]
    	for line in codecs.open(filepath,encoding="utf-8"):
            line = re.sub(ur'[a-zA-Z0-9\n!?<>\[\]:/~()\\@#$%&*\-,\."\'०१२३४५६७८९॥।]'," \n", line).split(' ') # custom expressions to be removed
            for word in line:
                if word !=('').encode("utf-8"):
                    corpus.append(word)
        corpus.append(u"\n")     # insert \n at start
        corpus.insert(0, u"\n")  # insert \n at end        
        return corpus
        
    
    def corp_dict(self, corpus):
        '''
        Build corpus dictionaries-
        cword2ind: {word:unique index in corpus}
        index2cwrd: {unique index in corpus: word}
        '''
    	wordset = sorted(set(corpus))
        cwrd2index = dict((c, i) for i, c in enumerate(wordset))
        index2cwrd = dict((i, c) for i, c in enumerate(wordset))
        return cwrd2index, index2cwrd 	    


    def load_vocab(self, vocab_path):
        '''
        Load word2vec model and build dictionaries

        word2ind: {word:unique index in word2vec vocab}
        index2wrd: {unique index in word2vec vocab: word}
        '''
        with open(vocab_path, 'r') as f:
            data = json.loads(f.read())
        ind2word = dict([(voc, key) for key, voc in data.items()])
        word2ind = data
        return ind2word, word2ind


    def add2ind(self, corpus, word2ind):
        '''
        Make dict for words which are not in word2vec vocab i.e out of vocabulary words
        '''
        add_wrd = []
        for word in corpus:
            if word not in word2ind:
                add_wrd.append(word)
     
        i = len(self.model.vocab)
        add_wrd = set(add_wrd)
        add_dict = {}
        for j in add_wrd:
            add_dict[j] = i
            i += 1
        return add_dict
    
    
    def corp2ind(self, corpus, full_dict):
        '''
        Vectorize corpus: map word to index using full_dict
        full_dict=word2ind+add_dict
        '''
        vec = []
        for word in corpus:
            ind = full_dict[word]
            vec.append(ind)
        return vec
    

    def prep_embed(self, full_i2w_dict, ind2word, w2v_dim):
        '''
        Prepare embedding vector for each word in full_dict
        
        Words which are in word2vec vocab are replaced by respective wordvector
        OOV words i.e words that are not in word2vec are replaced by random weight(rand_weight)
        '''
    	embed_weight=np.zeros((len(full_i2w_dict),w2v_dim))
        embed_dict={}
        for k,v in full_i2w_dict.items():
            if k in ind2word:
                model_weight=np.array(self.model[v])
                embed_weight[k]=model_weight
                embed_dict[k]=model_weight
            else: 
                rand_weight=np.array(glorot_uniform((w2v_dim,)).eval())
                embed_weight[k]=rand_weight
                embed_dict[k]=rand_weight
        return embed_weight, embed_dict


    def one_hot(self, ind, vocab_size):
        '''
        Prepare posiiton based one hot encoding
        '''
        empvec = np.zeros((vocab_size), dtype=np.bool)
        empvec[ind] = 1
        return empvec

    def y2vec(self, cwrd2index, y, vocab_size):
        '''
        Special function to map corpus to word indices using corpus dictionaries
        '''
        modvec=np.zeros((len(y),vocab_size))
        j=0
        for i in y:
            modvec[j]=self.one_hot(cwrd2index[i], vocab_size)
            j+=1
        return modvec


    def vec2seq(self, vec, seq_length):
        '''
        Reshape prepared vectors according to sequence length(seq_length)

        Output: 3D array 0f shape [no. of batches, seq_length, timestep]
        '''
        dim= vec.shape[0]/seq_length
        vec = np.reshape(vec, (dim, seq_length, vec.shape[1])).astype("int32")
        return vec

    
    def make_data(self,vec,seq_length,step,corpus):
        '''
        Special function for text prediction. Divides input into data sequence, labels

        Input: X=[[a,b,c],[b,c,d]] 
        Output: y=[d,e] 
        '''
        X_train=[]
        y_train=[]
        for i in range(0, len(vec)-seq_length, step):
            X_train.append(vec[i:i+seq_length])
            y_train.append(corpus[i+seq_length])
        return np.array(X_train), y_train
     
    
    def replace_oov(self,corpus,word2ind,oov_token):
        '''
        Replace OOV words with OOV token 
        '''
        nw_corpus=[]
        for word in corpus:
            if word not in word2ind:
                nw_corpus.append(oov_token)
            else:
                nw_corpus.append(word)    
        return nw_corpus
