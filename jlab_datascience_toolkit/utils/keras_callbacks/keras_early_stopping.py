from tensorflow import keras

class KerasEarlyStopping(object):
    
    # Initialize:
    #****************************
    def __init__(self,config):
        self.monitor = config.get("early_stopping_monitor",None)
        self.min_delta = config.get("early_stopping_min_delta",-1.0)
        self.patience = config.get("early_stopping_patience",0)
        self.verbose = config.get("early_stopping_verbose",0)
        self.mode = config.get("early_stopping_mode",'auto')
        self.baseline = config.get("early_stopping_baseline",None)
        self.best_weights = config.get("early_stopping_restore_best_weights",False)
        self.start_epoch = config.get("early_stopping_start_epoch",0)
    #****************************
    
    # Provide the callback
    #****************************
    def get_callback(self):
        if self.monitor is None or self.min_delta < 0.0:
            return None
        else:
            return keras.callbacks.EarlyStopping(
                monitor=self.monitor,
                min_delta=self.min_delta,
                patience=self.patience,
                verbose=self.verbose,
                mode=self.mode,
                baseline=self.baseline,
                restore_best_weights=self.best_weights,
                start_from_epoch=self.start_epoch
            )
    #****************************