import logging
import keras


trainer_log = logging.getLogger("Trainer Logger")


class Trainer:
    def __init__(self, configs: dict):
        '''
        Arguments of keras.Model.fit are divided into:

        1. Configurations passed in "configs" of Trainer.__init__ method. These are:
            a) batch_size=None
            b) epochs=1
            c) verbose="auto"
            d) callbacks=[]
            e) validation_split=0.0
            f) shuffle=True
            g) class_weight=None ===> WARNING: If passed, make sure class indices match those in targets "y"
            h) initial_epoch=0
            i) steps_per_epoch=None
            j) validation_steps=None
            k) validation_batch_size=None
            l) validation_freq=1
        
        2. Data (not configurations) passed to Trainer.fit method. These are:
            a) x
            b) y
            c) validation_data
            d) sample_weight
        
        In addition to the list in (1.), two additional items are in "configs":
        1) "loss_configs"
        2) "optimizer_configs"
        '''

        self.settings = configs.copy()
        self.settings.pop('registered_name')

        # 1) Loss
        loss_configs = self.settings.pop('loss_configs')
        loss_type = loss_configs.pop('loss_type')
        if loss_type == 'CategoricalCrossentropy':
            self.loss = keras.losses.CategoricalCrossentropy(**loss_configs)
        elif loss_type == 'SparseCategoricalCrossentropy':
            self.loss = keras.losses.SparseCategoricalCrossentropy(**loss_configs)
        else:
            raise NameError(f'Unrecognized loss_type ({loss_type}) !!!')
        
        # 2) Optimizer
        optimizer_configs = self.settings.pop('optimizer_configs')
        optimizer_type = optimizer_configs.pop('optimizer_type')
        if optimizer_type == 'Adam':
            self.optimizer = keras.optimizers.Adam(**optimizer_configs)
        elif optimizer_type == 'RMSprop':
            self.optimizer = keras.optimizers.RMSprop(**optimizer_configs)
        else:
            raise NameError(f'Unrecognized optimizer_type ({optimizer_type}) !!!')

        # 3) OPTIONAL Callbacks
        callbacks = []
        for callback_configs in self.settings.get('callbacks', []):
            callback_configs = callback_configs.copy()
            callback_type = callback_configs.pop('callback_type')
            if callback_type == 'EarlyStopping':
                callbacks.append(keras.callbacks.EarlyStopping(**callback_configs))
            elif callback_type == 'ReduceLROnPlateau':
                callbacks.append(keras.callbacks.ReduceLROnPlateau(**callback_configs))
            else:
                raise NameError('Unrecognized callback_type !!!')
        self.settings['callbacks'] = None if len(callbacks) == 0 else callbacks

        # 4) Check on "class_weight"
        if self.settings.get('class_weight', None) is not None:
            trainer_log.warning('Make sure indices of classes in "class_weight" match indices of "y" !')
    

    def fit(self, model, x=None, y=None, validation_data=None, sample_weight=None):
        model.model.compile(optimizer=self.optimizer, loss=self.loss)
        history = model.model.fit(
            x=x, y=y, validation_data=validation_data, sample_weight=sample_weight, **self.settings
        )
        return history