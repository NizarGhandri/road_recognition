from abc import ABC, abstractmethod
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from metrics import *
import torch 


class Model(ABC):


    def __init__ (self, model, history, loaded_trained, gpu):
        self.model = model
        self.history = history
        self.loaded_trained = loaded_trained
        if gpu: 
            if torch.cuda.is_available():
                torch.device("cuda")
            else: 
                raise("you don't have cuda please set it up and try again")
        else: 
            torch.device("cpu")


    @abstractmethod
    def build_model(self): 
        pass
    

    def serialize(self, path): 
        """
        serialize our model
        """
        if not os.path.isdir(path): 
            raise ValueError("path is incorrect")
        else:
            dir_name = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
            n_path = os.mkdir(os.path.join(path, dir_name))
            self.model = models.load_model(os.path.join(n_path, "model"))
            with open(os.path.join(n_path, "history"), 'wb') as file_pi:
                pickle.dump(self.history, file_pi)

    
    def load_serialized(self, path):
        """
        load an already serialized model
        """
        if not os.path.isdir(path): 
            raise ValueError("path is incorrect, please give a path for a directory")
        else:
            self.model = models.load_model(os.path.join(path, "model")) 
            self.loaded_trained = True
            if ("history.p" in os.listdir()):
                with open(os.path.join(path, "history.p"), 'rb') as file_pi:
                    self.history = pickle.load(history.history, file_pi)
                return True
            return False
    
    def predict(self, X):
        """
        after fitting our model, we used to predict the output of the testing set
        """
        if (not self.loaded_trained): 
            raise ValueError("Train or load a model before prediction")
        return self.model.predict(X) 


    def plot_history (self):
        """
        plot the accuracy, the F1 score, the precision and the recall as a function of the number of epochs
        """
        if not self.loaded_trained: 
            raise ValueError("Train or load a model beforehand")
            if self.history is None: 
                raise ValueError("History unavailable")
        # list all data in history
        print(self.history.history.keys())
        # summarize history for accuracy
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(history.history['acc'])
        axs[0, 0].plot(history.history['val_accuracy'])
        axs[0, 0].title('model accuracy')
        axs[0, 0].ylabel('accuracy')
        axs[0, 0].xlabel('epoch')
        axs[0, 0].legend(['train', 'validation'], loc='upper left')
        # summarize history for f1 score
        axs[0, 1].plot(history.history['f1_m'])
        axs[0, 1].plot(history.history['val_f1_m'])
        axs[0, 1].title('model f1 score')
        axs[0, 1].ylabel('f1 score')
        axs[0, 1].xlabel('epoch')
        axs[0, 1].legend(['train', 'validation'], loc='upper left')
        # summarize history for precision
        axs[1, 0].plot(history.history['precision_m'])
        axs[1, 0].plot(history.history['val_precision_m'])
        axs[1, 0].title('model precision')
        axs[1, 0].ylabel('precision')
        axs[1, 0].xlabel('epoch')
        axs[1, 0].legend(['train', 'validation'], loc='upper left')
        # summarize history for recall 
        axs[1, 1].plot(history.history['recall_m'])
        axs[1, 1].plot(history.history['val_recall_m'])
        axs[1, 1].title('model recall')
        axs[1, 1].ylabel('recall')
        axs[1, 1].xlabel('epoch')
        axs[1, 1].legend(['train', 'validation'], loc='upper left')
        plt.show()

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y), f1_m(y, predictions), precision_m(y, predictions), recall_m(y, predictions)

    def _callbacks_cnn(self,model_path='saved_model/cnn'):
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

    def fit (self, X, Y, epochs=600, batch_size=16, class_weights = None, plots=True) : 
    """
    We fit our model
    """
    assert(X.shape[1:] == self.input_shape)
    self.history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, \
                                use_multiprocessing=True, workers = os.cpu_count(),callbacks=self._callbacks_cnn())

    self.loaded_trained = True
    # list all data in history
    if (plots):
        print(self.history.history.keys())
        self.plot_history()
    return self.model


