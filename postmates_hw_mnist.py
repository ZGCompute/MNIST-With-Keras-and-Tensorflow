# file: postmates_hw_mnist.py
# Description: 
#
# This file contains the mnist_pipeline class for trainining a simple 
# 3 layer CNN with 2 fully connected layers and max pooling for downsampling. 
# The class also provides a method for performing inference on unseen images. 
# To enable large-scale experiments with adequate performance, a batch iterator 
# is used with image augmentation, as well as the tensorflow MirroredStrategy
# GPU method for parallelization. To enable automatic hyperparameter tuning, 
# a grid search method is employed.
#
# Arguments:
# 
#     use_gpu: Command-line option to specify gpu use
#              for accellerating training/inference. 
#              default is set to False
#
#     tune_params: Command-line option to specify mode,
#              as in hyperparameter tuning mode, or 
#              training mode, with already tuned or 
#              default params.
# 
#     serve: Command-line argument to specify serving
#              model inference outputs with frozen
#              model results. User must also specify
#              @model_path, and @image_path 
#
#     image_path: Command-line argument for specifying
#              path of an image for inference or serving
#              invference using frozen wieghts 
#
#     model_apth: Command-line argument for specifying
#              the path of a saved model to use for serving
#              or inference.
#     
# 
# Usage: python postmates_hw_mnist.py use_gpu tune_params serve predict model_path image_path
# Author: Zachary Greenberg
# Date last edited: 11/04/18

import os
import numpy as np
import functools
import requests
import json
from time import time
import argparse

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, \
AveragePooling2D, ZeroPadding2D, Flatten, Activation, Concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

import tensorflow as tf

class mnist_pipeline():
    '''Class mnist_pipeline provides methods for training with automated 
       hyper parameter tuning, parallelization, and performing inference on 
       MNIST benchmark data using a tensorflow server'''

    def __init__(self, use_gpu, tune_params, model_path):
        self.use_gpu = False
        if (use_gpu == 'True'):
            self.NUM_GPUS=4;
            self.use_gpu=True
        self.batch_size=10
        self.learn_rate = 0.001
        self.momentum= 0.9
        self.optimizer='SGD'
        self.num_epochs=1
        self.num_classes=10
        self.num_examples=6000
        self.saved_model_path= model_path
        if (os.path.exists(self.saved_model_path) == False):
                 os.mkdir(self.saved_model_path)
        self.accs = [] #store results of hparam tuning
        self.hparam_tune=tune_params
        self.dataset='mnist'
      
    def load_create_dataset(self):
        '''Load in the preferred data set to use for training/tuning.
           Return batch iterators for train/val'''

        if (self.dataset == 'mnist'):  
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if (self.dataset == 'cifar10'):
           (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)                                                                                                                                      
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)                                                                                                                                         
                                                                                                                                      
        x_train = x_train.astype('float32')                                                                                      
        x_test = x_test.astype('float32')                                                                                        
                                                                                                                                                                                    
        y_train = keras.utils.to_categorical(y_train, 10)                                                                                                                             
        y_test = keras.utils.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test);

    def get_model(self, learn_rate=0.01, momentum=0.9, optimizer='SGD'):
        '''Build a simple 3 layer CNN to fit on MNIST'''
        input_tensor = Input(shape=(28,28,1))
        x = Conv2D(32, (3, 3), padding='same', name='conv1', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x1 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
        x2 = MaxPooling2D(pool_size=(2, 2))(x2)

        x1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(x1)
        #x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(x2)
        #x2 = MaxPooling2D(pool_size=(2, 2))(x2)

        concat = Concatenate(axis=-1) 
        res = concat([x1, x2])
        res = Flatten()(res)

        res_fc = Dense(1000, activation='relu', name='fc1')(res)
        res_fc = Dense(500, activation='relu', name='fc2')(res)

        predictions = Dense(self.num_classes, activation='softmax', name='x_train_out')(res_fc)

        top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
        top3_acc.__name__ = 'top3_acc'

        model = Model(inputs=[input_tensor], outputs=predictions)
        
        if (optimizer=='SGD'): 
            optimizer=SGD(lr=learn_rate, momentum=momentum)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['mae','accuracy', top3_acc]) 

        #check if GPU mode enabled
        if (self.use_gpu==True):
             strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=self.NUM_GPUS)
             config = tf.estimator.RunConfig(train_distribute=strategy)
             estimator = tf.keras.estimator.model_to_estimator(model, config=config)
              
             return estimator

        return model

    def train(self):
        '''Tune Hparams and fit best model on dataset'''
        train, test = self.load_create_dataset()

        # batch generator with image augmentation
        train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.2,
                        horizontal_flip=True,
                        rotation_range=10.,
                        width_shift_range=0.2,
                        height_shift_range=0.2)

        test_datagen = ImageDataGenerator(rescale=1./255)
        train_datagen.fit(train[0]);
        test_datagen.fit(test[0]);

        #Hparams grid to search
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100, 500, 1000]
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        momentum = [0.2, 0.4, 0.6, 0.8, 0.9]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax', 'Nadam']

        if (self.hparam_tune==True):

            accs=[]
            for l in learn_rate: #grid search on lr and mtm
                for m in momentum:
                    model = get_model(learn_rate=l, momentum=m)
                    history_callback = model.fit_generator(generator=train_datagen.flow(train[0], train[1]), \
                                                           validation_data=test_datagen.flow(test[0], test[1]))
                    accs.append(history_callback.history["acc"])
                    if (np.argmax(accs) == len(accs)): #update hparams if recent is best
                       self.learn_rate = l
                       self.momentum= m
            accs=[]
            for b in batch_size:
                model = get_model()
                history_callback = model.fit_generator(generator=train_datagen.flow(train[0], train[1]), \
                                                       validation_data=test_datagen.flow(test[0], test[1]), steps_per_epoch=b)
                accs.append(history_callback.history["acc"])
                if (np.argmax(accs) == len(accs)):
                   self.batch_size = b

            accs=[]
            for e in epochs:
                model = get_model()
                history_callback = model.fit_generator(generator=train_datagen.flow(train[0], train[1]), \
                                                       validation_data=test_datagen.flow(test[0], test[1]), epochs=e)
                accs.append(history_callback.history["acc"])
                if (np.argmax(accs) == len(accs)):
                   self.num_epochs = e

            accs=[]
            for opt in optimizer:
                model = get_model(optimizer=opt)
                history_callback = model.fit_generator(generator=train_datagen.flow(train[0], train[1]), \
                                                       validation_data=test_datagen.flow(test[0], test[1]))
                accs.append(history_callback.history["acc"])
                if (np.argmax(accs) == len(accs)):
                   self.batch_size = b               

            # print tuning results
            print("Best: learning rate %f with momentum %f" % (self.learn_rate, self.momentum))
            print("Best: optimizer: %s and batch_size %d" % (self.optimizer, self.batch_size))
            print("Best: number of epochs: %d" % (self.num_epochs))      

        else: #train with default or tuned params
 
             #callbacks class to pass for saving shackpoints/early stopping
             class MyCbk(keras.callbacks.Callback):
    
                 def __init__(self, model):
                     self.model_to_save = model

                 def on_epoch_end(self, epoch, logs=None):
                     self.model_to_save.save('./model_at_epoch_%d.h5' %(epoch))

                 def chkPointer( self ):
                     ouput_file = '_mnist_model'
                     self.checkpointer = ModelCheckpoint(filepath=('./' + 'best' + output_file + ".hdf5"), verbose=1, save_best_only=True) 

                 def earlyStopeer( self ):
                     self.earlyStopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1) 

             #  use batch generator for training on large data set
             model = self.get_model(self.learn_rate, self.momentum, self.optimizer)
             cbk = MyCbk(model)
             tensorboard = TensorBoard(log_dir="~/Desktop/logs/{}".format(time()))
             history_callback = model.fit_generator(generator=train_datagen.flow(train[0], train[1]), validation_data=test_datagen.flow(test[0], test[1]), \
                                                    steps_per_epoch=self.batch_size, epochs=self.num_epochs, \
                                                    callbacks=[cbk, tensorboard])
             #save frozen weights
             model.save_weights('mnist_best_model.h5', self.saved_model_path)
             print(model.output.op.name) 

             #save for tensorflow server
             saver = tf.train.Saver()
             saver.save(K.get_session(), '%smnist_keras_model.ckpt' %(self.saved_model_path)) 
             os.system('python /usr/local/lib/python2.7/dist-packages/tensorflow/python/tools/freeze_graph.py \
                        --input_meta_graph=%smnist_keras_model.ckpt.meta \
                        --input_checkpoint=%smnist_keras_model.ckpt \
                        --output_graph=%skeras_frozen.pb \
                        --output_node_names="x_train_out/Softmax" \
                        --input_binary=true' %(self.saved_model_path,self.saved_model_path, self.saved_model_path))

    def predict(self, image):
        '''Run inference with saved model'''
        model = self.get_model(self.learn_rate, self.momentum, self.optimizer)          
        model.load_weights((self.saved_model_path + '/mnist_best_model.h5'))

        image = img_to_array(load_img(image, target_size=(28,28, 1))) / 255.
        preds = model.predict(image)
        print("Input image precidted as class: %d" %(np.argmax(preds)+1))

    def serve(self, image):
        '''Setup tensorflow server for requesting inference on sample images'''
        # initialize server from saved model and print info
        os.system('saved_model_cli show --dir %s --all' %(self.saved_model_path + 'keras_frozen.pb'))
        os.system("tensorflow_model_server --model_base_path=%s --rest_api_port=9000 \
                                           --model_name=MNISTModel" %(self.saved_model_path + 'keras_frozen.pb'))  

        image = img_to_array(load_img(image, target_size=(28,28, 1))) / 255.
        payload = {"instances": [{'input_image': image.tolist()}]}
        r = requests.get('http://localhost:9000/%s/MNISTModel:predict' %(self.saved_model_path + 'keras_frozen.pb'), json=payload)

        json.loads(r.content) 
        
if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('use_gpu', default=False)
    parser.add_argument('tune_params', default='False')
    parser.add_argument('serve', default='False')
    parser.add_argument('predict', default='False')
    parser.add_argument('image_path', default=None)
    parser.add_argument('model_path', default=None)
    args = parser.parse_args()

    mnist_pipe = mnist_pipeline(args.use_gpu, args.tune_params, args.model_path)
    mnist_pipe.train()
    
    if (args.serve=='True'):
        mnist_pipe.serve(args.image_path)

    if (args.predict=='True'):
        mnist_pipe.predict(args.image_path)          
         

        
