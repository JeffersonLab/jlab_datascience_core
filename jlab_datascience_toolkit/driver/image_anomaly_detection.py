import os
from jlab_datascience_toolkit.utils.graph_driver_utils import GraphRuntime
import numpy as np

modules = {
    'data_parser':'MNISTDataParser_v0',
    'data_scaler':'NumpyLinearScaler_v0',
    'anomaly_detector':'KerasCNNAE_v0',
    'loss_visualizer':'LearningCurveVisualizer_v0',
    'data_reconstruction':'DataReconstruction_v0',
    'residual_analysis':'ResidualAnalyzer_v0'
}

graph = [
    (None,'data_parser.load_data','mnist_data'),
    ('mnist_data','data_scaler.run','scaled_data'),
    ('scaled_data','anomaly_detector.train','training_history'),
    (None,'anomaly_detector.get_model','model'),
    ('training_history','loss_visualizer.run',None),
    (('scaled_data','model'),'data_reconstruction.run','rec_data'),
    ('rec_data','data_scaler.reverse','unscaled_data'),
    ('unscaled_data','residual_analysis.run','residuals')
]

this_file_loc = os.path.dirname(__file__)
cfg_locs = {
    'data_parser':os.path.join(this_file_loc,'../cfgs/defaults/mnist_data_parser_cfg.yaml'),
    'data_scaler':os.path.join(this_file_loc,'../cfgs/defaults/numpy_linear_scaler_cfg.yaml'),
    'anomaly_detector':os.path.join(this_file_loc,'../cfgs/defaults/keras_cnn_ae_cfg.yaml'),
    'loss_visualizer':os.path.join(this_file_loc,'../cfgs/defaults/learning_curve_visualizer_cfg.yaml'),
    'data_reconstruction':os.path.join(this_file_loc,'../cfgs/defaults/data_reconstruction_cfg.yaml'),
    'residual_analysis':os.path.join(this_file_loc,'../cfgs/defaults/residual_analyzer_cfg.yaml')
}

# Set the result location:
result_loc = 'anomaly_analysis_results_v0'

user_cfgs = {
    'data_parser':{},
    'data_scaler':{'A':1.0/255.0},
    'anomaly_detector':{
        'store_model_loc':result_loc,
        'image_dimensions':(28,28,1),
        'n_epochs': 25,
        'dense_architecture':[10,10],
        'dense_activations':['relu']*2,
        'dense_kernel_inits':['he_normal']*2,
        'dense_bias_inits':['he_normal']*2,
        'latent_space_is_2d':False,
        'optimizer':'legacy_adam',
        'early_stopping_monitor':'val_loss',
        'early_stopping_min_delta':0.00005,
        'early_stopping_patience':5,
        'early_stopping_restore_best_weights':True,
     },
     'loss_visualizer':{
         'output_loc':result_loc
     },
     'data_reconstruction':{},
     'residual_analysis':{
         'output_loc':result_loc,
         'real_data_name': 'x_orig'
        }
}

gr = GraphRuntime()
results, module_dict = gr.run_graph(graph, modules,cfg_locs,user_cfgs)
