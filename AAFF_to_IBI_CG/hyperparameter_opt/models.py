# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:11:03 2020

@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import os
import matplotlib.pyplot as plt
#import param
import sklearn
from sklearn.preprocessing import StandardScaler
#import data
import random
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def r2_metric(y_true, y_pred):
    SS_res =  ks.backend.sum(ks.backend.square(y_true - y_pred)) 
    SS_tot = ks.backend.sum(ks.backend.square(y_true-ks.backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + ks.backend.epsilon()) )

  
   


def shifted_sofplus(x):
    return ks.activations.softplus(x) - ks.backend.log(2.0)

def rmse(y_true, y_pred):
	return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true)))




#alpha=param.alpha
#beta=param.beta
  
def custom_activation(x):
    return (x**alpha)-(x**beta)



def getmodel(inpudim,sid):
    random.seed(sid)
    #Model
    activ = ks.layers.LeakyReLU(alpha=0.05)
    #activ = ks.layers.sigmoid()
    #activ=tf.keras.activations.sigmoid()

    depth = random.randint(1, 5)  #depth of NN
    geo_input = ks.Input(shape=(inpudim,), dtype='float32' ,name='geo_input')
    #geo_input = ks.Input(shape=(inpudim), dtype='float32' ,name='geo_input')
    pre = geo_input
    #l_rate=random.uniform(1,4)
    #learning_rate=10**(-l_rate)
    regu=random.uniform(1,6)
    regulization=10**(-regu)
#    l_rate=random.uniform(1,6)
#    learning_rate=10**(-l_rate)
#    learning_rate_stop = 5e-5
    M=random.randint(10, 1000)
    # Some Pre-processing here...
    full = ks.layers.Flatten()(pre)
    full =  ks.layers.Dense(M,
                                use_bias=True,
                                activation=activ,
                                kernel_regularizer = tf.keras.regularizers.l2(regulization)
                                #kernel_regularizer = tf.keras.regularizers.l1(1e-3)
                                )(full)
    k=random.uniform(0.1,1)
    layer=[]
    for i in range(depth):
        j=(i+1)
        neuron=int((k**(j))*M)
        if neuron<10:
            neuron=10
        layer.append(neuron)

        full =  ks.layers.Dense(neuron,
                                use_bias=True,
                                activation=activ,
                                kernel_regularizer = tf.keras.regularizers.l2(regulization)
                                #kernel_regularizer = tf.keras.regularizers.l1(1e-3) 
                                )(full)
    main_output =  ks.layers.Dense(900,name='main_output',use_bias=True,kernel_regularizer = tf.keras.regularizers.l2(regulization),activation='linear')(full)
    #main_output =  ks.layers.Dense(1,name='main_output')(full)
    model = ks.models.Model(inputs=geo_input, outputs=main_output)
    print ("depth=",depth,"FN=",M, "Neoron=",layer, "Regulization=",regulization)
    return model



def model_training(model_name,model,xtrain,ytrain,xtest,ytest,sid):
    random.seed(sid)
    l_rate=random.uniform(1,6)
    learning_rate=10**(-l_rate)
    learning_rate_stop = 5e-5
    print ("Learning_Rate =", learning_rate)
    epo = 200
    epostep = 1
    #batch_size = param.batch_size
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    #mae=get_MAE(y_scaler)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error',rmse,lr_metric,r2_metric])

    cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate - (learning_rate-learning_rate_stop)/epo*epoch)]


    #hist = model.fit(xtrain, ytrain, epochs=epo,callbacks=cbks,validation_freq=epostep,validation_data=(xtest,ytest),verbose=2)
    hist=model.fit(xtrain, ytrain, epochs=epo,callbacks=cbks,validation_freq=epostep,validation_data=(xtest,ytest),verbose=2)
   #hist = model.fit(xtrain, ytrain, epochs=epo,batch_size=batch_size,callbacks=cbks,validation_data=(xval,yval),verbose=2)
    trainlossall = hist.history['mean_absolute_error']
    testlossall = hist.history['val_mean_absolute_error']

    #Save training info
    trainlossall = np.array(trainlossall)
    testlossall = np.array(testlossall)
    #if(save_weights == True):
    #model.save_weights(os.path.join("data",model_name + '_model_weights.h5'))
    #np.save(os.path.join(outdir,model_name + '_trainloss.npy'),trainlossall)
    #np.save(os.path.join(outdir,model_name + '_testloss.npy'),testlossall)


    #Plot training results
    #print(model.summary())

    y_pred_train = model.predict(xtrain)

    y_pred_test = model.predict(xtest)
    return y_pred_train, y_pred_test
















