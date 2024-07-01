from jlab_datascience_toolkit.core.jdst_model import JDSTModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from evms_ultra_dev.utils.architectures.keras_cnn_ae_architecture import KerasCNNAEArchitecture
from evms_ultra_dev.utils.keras_callbacks.keras_early_stopping import KerasEarlyStopping

class KerasCNNAE(keras.Model,JDSTModel):
    '''
    Class for setting up an Autoencoder with convolutional and optionally dense layers. This class uses the KerasCNNArchitecture class
    to set up the architecture.
    '''

    # Initialize:
    #****************************
    def __init__(self,path_to_cfg,shape,user_config={}):
        super(KerasCNNAE, self).__init__()

        # Set the name specific to this module:
        self.module_name = "keras_cnn_ae"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Retreive settings from configuration:
        precision = self.config['precision']
        # Get the architecture class:
        self.ae_architecture = KerasCNNAEArchitecture(precision)
        
        # Model storage / loading and model format:
        self.load_model_loc = self.config['load_model_loc']
        self.model_store_format = self.config['model_store_format']
        self.compile_loaded_model = self.config['compile_loaded_model']

        # Get the image dimensions --> Important for setting the network architecture properly:
        self.image_dims = (shape[1],shape[2],shape[3])

        # NETWORK ARCHITECTURE AND FEATURES:

        # Down / Up-sampling of the image, in case it is too large:
        max_pooling = self.config['max_pooling']
        # Convolutional units:
        conv_architecture = self.config['conv_architecture']
        conv_activations = self.config['conv_activations']
        conv_kernel_inits = self.config['conv_kernel_inits']
        conv_bias_inits = self.config['conv_bias_inits']
        kernel_sizes = self.config['kernel_sizes']
        strides = self.config['strides']

        # Run consistency check on strides and maximm pooling:
        strides, d_shape_x, d_shape_y = self.inspect_pooling_and_strides(self.image_dims,max_pooling,strides)

        # Dense units:
        dense_architecture = self.config['dense_architecture']
        dense_activations = self.config['dense_activations']
        dense_kernel_inits = self.config['dense_kernel_inits']
        dense_bias_inits = self.config['dense_bias_inits']

        # Latent dimension:
        latent_dim = self.config['latent_dim']
        latent_space_is_2d = self.config['latent_space_is_2d']

        # Handle reshaping the inputs for the decoder (going from flat to Conv2D):
        decoder_conv_reshape_units = self.config['decoder_conv_reshape_units']
        decoder_conv_reshape = [int(d_shape_x),int(d_shape_y),decoder_conv_reshape_units]

        # Response of decoder output layer:
        output_activation = self.config['output_activation']
        output_filter = shape[3]
        output_kernel_size = self.config['output_kernel_size']
        output_strides = self.config['output_strides']

        # OPTIMIZER AND LOSS FUNCTION:
        self.learning_rate = self.config['learning_rate']
        self.optimizer_str = self.config['optimizer']
        self.loss_function_str = self.config['loss_function']

        # Make sure that decoder outputs are set properly, if we decide to work with logits:
        self.use_logits = False 
        if self.loss_function_str.lower() == "logit_bce":
            output_activation = "linear"
            self.use_logits = True

        # TRAINING:
        self.n_epochs = self.config['n_epochs']
        self.batch_size = self.config['batch_size']
       
        # Add early stopping callback (if config is properly set):
        self.early_stopping = KerasEarlyStopping(self.config)

        # BUILD THE MODEL:
        # Check if the model already exists and just needs to be loaded:
        if self.load_model_loc is not None:
            self.load(self.load_model_loc)
        else:
            # Encoder:
            self.encoder = self.ae_architecture.get_encoder(
               input_dimensions=self.image_dims,
               conv_architecture=conv_architecture,
               conv_activations=conv_activations,
               conv_kernel_inits=conv_kernel_inits,
               conv_bias_inits=conv_bias_inits,
               kernel_sizes=kernel_sizes,
               strides=strides,
               dense_architecture=dense_architecture,
               dense_activations=dense_activations,
               dense_kernel_inits=dense_kernel_inits,
               dense_bias_inits=dense_bias_inits,
               latent_dim=latent_dim,
               latent_activation='linear',
               latent_kernel_init='glorot_normal',
               latent_is_2d=latent_space_is_2d,
                max_pooling=max_pooling,
                encoder_name="Encoder"
            )

            # Decoder:
            self.decoder = self.ae_architecture.get_decoder(
               latent_dim=latent_dim,
               latent_is_2d=latent_space_is_2d,
               reshape_dimensions=decoder_conv_reshape,
               conv_architecture=conv_architecture[::-1],
               conv_activations=conv_activations[::-1],
               conv_kernel_inits=conv_kernel_inits[::-1],
               conv_bias_inits=conv_bias_inits[::-1],
               kernel_sizes=kernel_sizes[::-1],
               strides=strides[::-1],
               dense_architecture=dense_architecture[::-1],
               dense_activations=dense_activations[::-1],
               dense_kernel_inits=dense_kernel_inits[::-1],
               dense_bias_inits=dense_bias_inits[::-1],
               output_filter=output_filter,
               output_kernel_size=output_kernel_size,
               output_strides=output_strides,
               output_activation=output_activation,
               max_pooling=max_pooling,
               decoder_name="Decoder"
            )

            # Compile the model:
            self.compile()
    #****************************

    # Run dimensional check on max. pooling and strides       
    #****************************
    def inspect_pooling_and_strides(self,idims,max_pooling,stride_list):
        new_stride_list = []
        xdim = idims[0]
        ydim = idims[1]
        
        # Apply max. pooling
        if max_pooling is not None and max_pooling[0] > 0 and max_pooling[1] > 0:
            xdim /= max_pooling[0]
            ydim /= max_pooling[1]

        #++++++++++++++++++++
        for s in stride_list:
            t_x = xdim % s
            t_y = ydim % s

            if t_x == 0 and t_y == 0:
                xdim /= s
                ydim /= s
                new_stride_list.append(s)
            else:
                new_stride_list.append(1)
        #++++++++++++++++++++
                
        return new_stride_list, xdim,ydim
    #****************************

    # Compile the model:
    #****************************
    def compile(self):
        super(KerasCNNAE, self).compile()
        # Register the components:
        
        # Specify optimizer and loss function:
        self.optimizer = None
        self.loss_fn = None
        
        if self.optimizer_str.lower() == "adam":
            self.optimizer = keras.optimizers.Adam(self.learning_rate)

        if self.optimizer_str.lower() == "legacy_adam":
            self.optimizer = keras.optimizers.legacy.Adam(self.learning_rate)

        if self.optimizer_str.lower() == "sgd":
            self.optimizer = keras.optimizers.SGD(self.learning_rate)

        if self.loss_fn_str.lower() == "mse":
            self.loss_fn = keras.losses.MeanSquaredError()

        if self.loss_fn_str.lower() == "mae":
            self.loss_fn = keras.losses.MeanAbsoluteError()

        if self.loss_fn_str.lower() == "huber":
            self.loss_fn = keras.losses.Huber()
        
        if self.loss_fn_str.lower() == "logit_bce":
            def loss_func(x,x_logit):
                return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x))
            
            self.loss_fn = loss_func
    #****************************

    # Get Model response:
    #****************************
    # Call:
    def call(self,x):
        z = self.encoder(x)
        return self.decoder(z)
    
    #-------------------------

    # Predict:
    def predict(self,x,to_numpy=True):
        z = self.encoder(x)
        x_rec = self.decoder(z)
 
        if self.use_logits == True:
            x_rec = tf.sigmoid(x_rec)

        gc.collect()

        if to_numpy == True:
            return{
                'z_model':z.numpy(),
                'x_rec':x_rec.numpy()
            }
        
        return{
                'z_model':z,
                'x_rec':x_rec
        }
    #****************************

    # Autoencoder training:
    #****************************
    # Train step:
    @tf.function
    def train_step(self,x):
        with tf.GradientTape() as tape:
            z = self.encoder(x)
            x_rec = self.decoder(z)
            loss = self.loss_fn(x,x_rec)

        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        
        return {
            'loss':loss,
        }
    
    #-------------------------

    @tf.function
    def test_step(self,x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        loss = self.loss_fn(x,x_rec)

        return {
            'loss':loss
           }
        
    #-------------------------
    
    # Entire fit function:
    def train(self,x,n_epochs=None,batch_size=None,validation_split=None,verbosity=None):
        # Use the default settings, if no explicit ones are provided:
        if n_epochs is None:
            n_epochs = self.n_epochs

        if batch_size is None:
            batch_size = self.batch_size

        if validation_split is None:
            validation_split = self.validation_split

        if verbosity is None:
            verbosity = self.verbosity

        # Divide the data in training and validations data:
        x_train, x_test = train_test_split(x,test_size=validation_split)

        # Handle callbacks:
        ae_callbacks = [tf.keras.callbacks.TerminateOnNaN(),GarbageHandler()]
        if self.early_stopping is not None:
            ae_callbacks.append(self.early_stopping)

        results = super(KerasCNNAE, self).fit(
              x=x_train,
              y=None,
              validation_data=(x_test,None),
              batch_size=batch_size,
              epochs=n_epochs,
              shuffle=True,
              callbacks=ae_callbacks,
              verbose=verbosity
        )

        outputs = {}
        #+++++++++++++++++++
        for key in results.history:
            outputs[key] = results.history[key]
        #+++++++++++++++++++
            
        return outputs
    #****************************



    # Store / load  the network:
    #****************************
    # Save the entire model:
    def save(self,model_loc):
        self.encoder.save(model_loc+"/keras_cnn_ae_encoder"+self.model_store_format)
        self.decoder.save(model_loc+"/keras_cnn_ae_decoder"+self.model_store_format)

    #----------------

    def load(self,model_loc):
        self.encoder = keras.models.load_model(model_loc+"/keras_cnn_ae_encoder"+self.model_store_format)
        self.decoder = keras.models.load_model(model_loc+"/keras_cnn_ae_decoder"+self.model_store_format)

        # Check if re-compilation is required:
        if self.compile_loaded_model == True:
            # We need to keep track of the weights, otherwise, they are lost after compilation:
            encoder_weights = self.encoder.get_weights()
            decoder_weights = self.decoder.get_weights()

            self.compile()

            self.encoder.set_weights(encoder_weights)
            self.decoder.set_weights(decoder_weights)
    #****************************
    
    
    