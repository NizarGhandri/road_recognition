from model import Model
import os 
import metrics
import matplotlib.pyplot as plt
import preprocess_data
from tensorflow import keras
from tensorflow.keras import datasets, layers, models,regularizers
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Conv2D,\
     MaxPool2D, Dropout, Input, AveragePooling2D, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



#pour le moment not called
def callbacks_cnn(model_path='saved_model/cnn'):
    """
    function to get checkpointer, early stopper and lr_reducer in our CNN
    """
    #Callback to save the Keras model or model weights at some frequency.
    checkpointer = ModelCheckpoint(model_path,
                                monitor=f1_m,
                                mode="max",
                                save_best_only = True,
                                verbose=1)

    #Stop training when f1_m metric has stopped improving for 20 epochs
    earlystopper = EarlyStopping(monitor =f1_m, 
                                mode='max', 
                                patience = 5,
                                verbose = 1,
                                restore_best_weights = True)

    #Reduce learning rate when loss has stopped improving for 3 epochs
    lr_reducer = ReduceLROnPlateau(monitor='loss',
                                mode='min',
                                factor=0.9,
                                patience=2,
                                min_delta= 0.001, 
                                min_lr=0.00001,
                                verbose=1)

    return [checkpointer, earlystopper, lr_reducer]

class Convolutinal_neural_net (Model):

    def __init__ (self, input_shape=None, build_on_init=True):
        if (build_on_init):
            if (input_shape is None):  
                raise ValueError("input_shape needs to be specified")
            else: 
                self.build_model()

        self.img_size=input_shape[0]
        self.n_channels=input_shape[2]

        self.input_shape = input_shape
        self.model = None
    
    def apply_convolution(self,input, n_filters=16, kernel_size=3, activation_Leaky=False ,\
        alpha=0.3 , dropout_rate=0.3 ,padding_type='same',pool_size_dropout=(2,2),strides_shape=(2, 2)):
        """
        function to do one whole concolutional layer
        1) Apply convolution
        2) Apply BAtch Normalization
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
        layer1 = self.apply_convolution(input=pool0, n_filters=64)
        layer2 = self.apply_convolution(input=layer1, n_filters=128,  activation_Leaky=True)
        layer3 = self.apply_convolution(input=layer2, n_filters=512,  activation_Leaky=True)

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
        
        #c quoi classweights? A CHANGEEEEEEEER
    def fit (self, X, Y, epochs=3, batch_size=64, class_weights = None, plots=True) : 
        assert(X.shape[1:] == self.input_shape)
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, class_weights = None,\
                                 use_multiprocessing=True, workers = os.cpu_count())
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(history.history['accuracy'])
        axs[0, 0].plot(history.history['val_accuracy'])
        axs[0, 0].title('model accuracy')
        axs[0, 0].ylabel('accuracy')
        axs[0, 0].xlabel('epoch')
        axs[0, 0].legend(['train', 'validation'], loc='upper left')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return self.model

    
        
