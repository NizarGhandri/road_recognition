from model import Model
import os 
import Metrics


class Convolutinal_neural_net (Model):

    def __init__ (self, input_shape=None, build_on_init=True):
        if (build_on_init):
            if (input_shape is None):  
                raise ValueError("if you want to build the model when you initialize you have to specify the shape otherwise set build_on_inint to False")
            else: 
                self.build_model()

        self.input_shape = input_shape
        self.model = None


    
    def build_model(self): 
        #Input layer
        INPUT = keras.layers.Input(shape=self.input_shape)

        pool0=keras.layers.AveragePooling2D(pool_size=(2,2),
                                            strides=(1,1))(INPUT)
                                            
        conv1 =keras.layers.Conv2D(filters=128,
                                kernel_size=(5,5),
                                strides=(1,1),
                                activation='relu',
                                padding='same',
                                input_shape=INPUT_SHAPE)(pool0)

        pool1 = keras.layers.MaxPool2D(pool_size=(2,2),
                                    padding='same')(conv1)
        drop1 = keras.layers.Dropout(0.2)(pool1)

        conv2 =  keras.layers.Conv2D(filters=256,
                                kernel_size=(5,5),
                                strides=(2,2),
                                activation='relu',
                                padding='same',
                                input_shape=INPUT_SHAPE)(drop1)    
        pool2 = keras.layers.MaxPool2D(pool_size=(2,2),
                                    padding='same')(conv2)
        drop2 = keras.layers.Dropout(0.3)(pool2)


        conv3 =  keras.layers.Conv2D(filters=64,
                                kernel_size=(10,10),
                                strides=(2,2),
                                activation='relu',
                                padding='same',
                                input_shape=INPUT_SHAPE)(drop2)  
                                
        pool3 = keras.layers.MaxPool2D(pool_size=(2,2),
                                    padding='same')(drop2)
        drop3 = keras.layers.Dropout(0.3)(pool3)

        flat = keras.layers.Flatten()(drop3)

        dense1 = keras.layers.Dense(32,
                                    kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                    activation='softmax')(flat)

        dense2 = keras.layers.Dense(2,
                                    kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                    activation='sigmoid')(dense1)

        self.model = keras.Model(inputs=INPUT,outputs=dense2)
        print(self.model.summary())
        self.model.compile(optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m], loss='categorical_crossentropy')
        return self.model
        
        
    def fit (self, X, y, epochs=3, batch_size=64, class_weights = None, plots=True) : 
        assert(X.shape[1:] == self.input_shape)
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, class_weights = class_weights, use_multiprocessing=True, workers = os.cpu_count())
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

        
