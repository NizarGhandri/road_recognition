from model import Model
from metrics import *
from tensorflow import keras
from tensorflow.keras import datasets, layers, models,regularizers
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Conv2D,\
     MaxPool2D, Dropout, Input, AveragePooling2D, Dense
import os


class Convolutional_neural_net (Model):

    def __init__ (self, input_shape=None, build_on_init=True, gpu=True):
        """
        Initialize attributes
        """

        super().__init__(None, None, False, gpu)
        self.set_input_shape((None, None, None))

        if (build_on_init):
            if (input_shape is None):  
                raise ValueError("input_shape needs to be specified if you want to build on initialization")
            else: 
                self.set_input_shape(input_shape)
                self.build_model()


    def set_input_shape (self, input_shape):
        """ 
        Input shape setter
        """
        self.input_shape = input_shape
        self.img_size=input_shape[0]
        self.n_channels=input_shape[2]

    
    def _apply_convolution(self, input, n_filters=16, kernel_size=3, activation_Leaky=False , alpha=0.3 , dropout_rate=0.3 ,padding_type='same',pool_size_dropout=(2,2),strides_shape=(2, 2)):
        """
        function to do one whole convolutional layer
        1) Apply convolution
        2) Apply Batch Normalization
        3) Apply activation function
        4) Apply maxpooling
        5) Apply Dropout
        """

        #first convolutional layer
        conv1 =  Conv2D(filters=n_filters,
                        kernel_size= kernel_size,
                        kernel_initializer= 'he_normal',
                        strides=strides_shape,
                        padding='same')(input)
        
        #batch normalization
        norm = BatchNormalization()(conv1)

        #activation function
        if activation_Leaky:
            act= LeakyReLU(alpha)(norm)
        else: 
            act= ReLU(alpha)(norm)

        #maxpooling
        pool=MaxPool2D(pool_size=pool_size_dropout, padding=padding_type)(act)

        #dropout : written as a variable for clarity
        drop=Dropout(dropout_rate)(pool)
        
        return drop

    def build_model(self): 
        #Input layer
        INPUT = Input((self.img_size, self.img_size , self.n_channels))

        #Apply average pooling for the input
        pool0=AveragePooling2D(pool_size=(2,2),strides=(1,1))(INPUT)

        #Apply three convolutional layers              
        layer1 = self._apply_convolution(input=pool0, n_filters=64)
        layer2 = self._apply_convolution(input=layer1, n_filters=128,  activation_Leaky=True)
        layer3 = self._apply_convolution(input=layer2, n_filters=512,  activation_Leaky=True)

        #Flatten
        flat = keras.layers.Flatten()(layer3)

        #Apply three dense layers
        dense1 = Dense(units=128,kernel_regularizer=regularizers.l1(0.01),
                                    activation='softmax')(flat)
        dense2 = Dense(units=32,kernel_regularizer=regularizers.l1(0.01),
                                    activation='softmax')(dense1)
        dense3 = Dense(units=2,kernel_regularizer=regularizers.l1(0.01),
                                    activation='sigmoid')(dense2)

        self.model = keras.Model(inputs=INPUT,outputs=dense3)
        print(self.model.summary())
        self.model.compile(optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m],\
                            loss='binary_crossentropy')
        return self.model


    def fit (self, X, Y, epochs=3, batch_size=64, class_weights = None, plots=True) : 
        """
        We fit our model
        """
        assert(X.shape[1:] == self.input_shape)
        print(os.cpu_count())
        self.history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, \
                                 use_multiprocessing=True, workers = os.cpu_count(),callbacks=self._callbacks_cnn())

        self.loaded_trained = True
        # list all data in history
        if (plots):
            print(self.history.history.keys())
            self.plot_history()
        return self.model



    
     
 
        
