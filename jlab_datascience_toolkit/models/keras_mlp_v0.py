import os
import yaml
import keras
import inspect
from jlab_datascience_toolkit.core.jdst_model import JDSTModel


class KerasMLP(JDSTModel):
    '''
    Defines an MLP model. self is not a keras.Model itself. Instead, it has a "model" attribute which is a keras.Model.
    '''
    def __init__(self, configs: dict):
        '''
        configs has the following keywords:
        1) 'input_dim'
        2) 'layers_dicts': List of subdictionaries, each subdictionary contains configs of one of:
            2.1) 'layer_type': 'Dense', 'layer_configs': keras Dense layer configs
            2.2) 'layer_type': 'Dropout', 'layer_configs': keras Dropout layer configs
            2.3) 'layer_type': 'BatchNormalization', 'layer_configs': keras BN layer cnfigs
        '''
        self.configs = configs
        inputs = keras.layers.Input(shape=(configs['input_dim'],))
        outputs = inputs
        for layer_dict in configs['layers_dicts']:
            layer_type = layer_dict['layer_type']
            layer_configs = layer_dict.get('layer_configs', {})
            if layer_type == 'Dense':
                outputs = keras.layers.Dense(**layer_configs)(outputs)
            elif layer_type == 'Dropout':
                outputs = keras.layers.Dropout(**layer_configs)(outputs)
            elif layer_type == 'BatchNormalization':
                outputs = keras.layers.BatchNormalization(**layer_configs)(outputs)
            else:
                raise NameError('Unrecognized layer_type !!!')
        self.model = keras.models.Model(inputs=inputs, outputs=outputs)

    def predict(self, x):
        y = self.model.predict(x)
        return y
    
    def get_info(self):
        """Prints this module's docstring."""
        print(inspect.getdoc(self))
    
    def load(self, folder_path: str):
        assert os.path.exists(folder_path)
        self.load_model(os.path.join(folder_path, 'model.keras'))
        loaded_configs = self.load_config(os.path.join(folder_path, 'configs.yaml'))
        assert self.configs == loaded_configs, 'Mismatch between configs with which model was instantiated and loaded configs !!!'
    
    def load_model(self, path: str):
        assert path.endswith('.keras')
        self.model = keras.models.load_model(path)
    
    @staticmethod
    def load_config(path: str):
        assert path.endswith('.yaml')
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    
    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        self.save_model(os.path.join(folder_path, 'model.keras'))
        self.save_config(os.path.join(folder_path, 'configs.yaml'))

    def save_model(self, path: str):
        assert path.endswith('.keras')
        self.model.save(path)
    
    def save_config(self, path: str):
        assert path.endswith('.yaml')
        with open(path, "w") as file:
            yaml.dump(self.configs, file)