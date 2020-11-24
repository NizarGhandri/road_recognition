from model import Model
from metrics import *
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models,regularizers
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Conv2D,\
     MaxPool2D, Dropout, Input, AveragePooling2D, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



class Convolutional_neural_net (Model):

    def __init__ (self, input_shape=None, build_on_init=True, gpu = True):
        """
        Initialize attributes
        """

        super().__init__(None, None, False)
        self.set_input_shape((None, None, None))
        self.gpu = gpu

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

    

    def _conv(input, n_filters, kernel_size, activation_Leaky , alpha ):

        #first convolutional layer
        conv1 =  Conv2D(filters=n_filters,
                        kernel_size= kernel_size,
                        kernel_initializer= 'he_normal',
                        padding='same')(input)
        
        #batch normalization
        conv1 = BatchNormalization()(conv1)

        #activation function
        if activation_Leaky:
            conv1= LeakyReLU(alpha)(conv1)
        else: 
            conv1= ReLU(alpha)(conv1)


        #second convolutional layer
        conv2= Conv2D(filters=n_filters, 
                        kernel_size= kernel_size, 
                        kernel_initializer=  'he_normal',
                        padding='same')(conv1)
        
        #batch normalization
        conv2 = BatchNormalization()(conv2)

        #activation function
        if activation_Leaky:
            conv2= LeakyReLU(alpha)(conv2)
        else :
            conv2= ReLU(alpha)(conv2)

        return conv2

    
    def _down_sample(conv , dropout_val ):

        pool = MaxPooling2D(pool_size=(2, 2) ,strides=None, padding='same', data_format='channels_last')(conv)

        if dropout_val != None: 
        pool = Dropout(dropout_val)(pool)

        return pool 

    def _up_sample(conv1, conv2, n_filters, kernel_size ): 

        up =Conv2DTranspose(n_filters, kernel_size= kernel_size, strides=(2, 2), padding='same') (conv1)

        merge = concatenate([conv2,up], axis = 3)

        return up, merge

    # U-net model

    def build_model(n_filters= 16, kernel_size= 3, activation_Leaky=True, dropout_val=0.5 , lr=0, alpha=0.3):
        input_imgs = Input((self.img_size, self.img_size , self.n_channels))

        """
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        """

        #encoder
        #1
        conv1= conv(input_imgs, n_filters*1, kernel_size, activation_Leaky, alpha )
        down1 = down_sample(conv1 , dropout_val)

        #2
        conv2= conv(down1, n_filters*2, kernel_size, activation_Leaky, alpha )
        down2 = down_sample(conv2 , dropout_val)

        #3
        conv3= conv(down2, n_filters*4,  kernel_size, activation_Leaky, alpha  )
        down3 = down_sample(conv3 , dropout_val)

        #4
        conv4= conv(down3, n_filters*8,  kernel_size, activation_Leaky , alpha )
        down4 = down_sample(conv4 , dropout_val)

        #5
        conv5= conv(down4, n_filters*16,  kernel_size, activation_Leaky, alpha  )
        drop5 = Dropout(dropout_val)(conv5)
        #drop5= conv5


        """
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        """

        

        """
        up6 = Conv2D(512, 2, activation , padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        """

        #decoder  
        up6, merge6= up_sample(drop5, conv4,  n_filters*8 , kernel_size) 
        conv6 = conv(merge6, n_filters*8 , kernel_size, activation_Leaky, alpha)

        """
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        """

        """
        up7 = Conv2D(256, 2, activation , padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        """

        up7, merge7= up_sample(conv6, conv3,  n_filters*4 , kernel_size )
        conv7 = conv(merge7, n_filters*4 , kernel_size, activation_Leaky, alpha )

        """
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        """

        """
        up8 = Conv2D(128, 2, activation , padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        """

        up8, merge8= up_sample(conv7, conv2,  n_filters*2 ,kernel_size )
        conv8 = conv(merge8,n_filters*2 , kernel_size, activation_Leaky, alpha )

        """
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        """
        """
        up9 = Conv2D(64, 2, activation , padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        """

        up9, merge9= up_sample(conv8, conv1,  n_filters*1 ,kernel_size )
        conv9 = conv(merge9, n_filters*1 , kernel_size, activation_Leaky, alpha )

        """
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        """

        #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        # 1 x 1 convolution
        output = Conv2D(1, 1, activation = 'sigmoid')(conv9)



        self.model = Model(inputs = [input_imgs], outputs = [output])

        self.model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['acc',f1_m, precision_m, recall_m])
        
        print(self.model.summary())


        return self.model


 



            


     
 
        
