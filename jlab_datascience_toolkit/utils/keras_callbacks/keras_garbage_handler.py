import tensorflow as tf
import gc

# Try to keep track of the memory consumption during the training phase:
# This code was taken from: 
# https://stackoverflow.com/questions/64666917/optuna-memory-issues
class KerasGarbageHandler(tf.keras.callbacks.Callback):

    # Clear memory at the end of every training epoch:
    #******************************
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
    #******************************



