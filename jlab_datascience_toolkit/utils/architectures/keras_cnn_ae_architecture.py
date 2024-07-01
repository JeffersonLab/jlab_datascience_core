import tensorflow as tf
from tensorflow import keras

class KerasCNNAEArchitecture(object):
    '''
    Specify the architecture of a (Variatonal) AutoEncoder
    '''

    # Initialize:
    #*************************
    def __init__(self,precision='float32'):
        self.precision = precision
    #*************************
        
    # Helper function to set / register activation functions:
    #*************************
    def get_activation_function(self,act_fn_str,name):
        if act_fn_str.lower() == "leaky_relu":
            return keras.layers.Activation(tf.nn.leaky_relu,name=name)
        
        return keras.layers.Activation(act_fn_str.lower(),name=name,dtype=tf.keras.mixed_precision.Policy(self.precision))  
    #*************************

    # Encoder:
    #*************************
    def get_encoder(self,input_dimensions,conv_architecture,conv_activations,conv_kernel_inits,conv_bias_inits,kernel_sizes,strides,dense_architecture,dense_activations,dense_kernel_inits,dense_bias_inits,latent_dim,latent_activation,latent_kernel_init,latent_is_2d,max_pooling,encoder_name):
        # Get the number of conv. / dense layers:
        n_conv_layers = len(conv_architecture)
        n_dense_layers = len(dense_architecture)
        
        # Define the encoder input:
        encoder_inputs = keras.layers.Input(
            shape=input_dimensions,
            name=encoder_name+"_input"
        )

        # Apply max-pooling, if requested:
        x_enc = encoder_inputs
        if max_pooling is not None and max_pooling[0] > 0 and max_pooling[1] > 0:
            x_enc = keras.layers.MaxPooling2D(
                pool_size=(max_pooling[0],max_pooling[1]),
                dtype=tf.keras.mixed_precision.Policy(self.precision)
            )(x_enc)

        # Add conv. parts:
        #++++++++++++++++++++++++++
        for k in range(n_conv_layers):
            conv_laver =  keras.layers.Conv2D(
                    filters=conv_architecture[k], 
                    kernel_size=kernel_sizes[k], 
                    strides=strides[k],
                    kernel_initializer=conv_kernel_inits[k],
                    bias_initializer=conv_bias_inits[k],
                    name=encoder_name+"_conv"+str(k),
                    dtype=tf.keras.mixed_precision.Policy(self.precision) 
            )
            conv_activation = self.get_activation_function(conv_activations[k],name=encoder_name+"_conv_act"+str(k))

            x_enc = conv_laver(x_enc)
            x_enc = conv_activation(x_enc)
        #++++++++++++++++++++++++++
        
        # Add a flattening layer:
        x_enc = keras.layers.Flatten(dtype=tf.keras.mixed_precision.Policy(self.precision))(x_enc)
        
        # Add a dense part, if wanted:
        if n_dense_layers > 0:
              #++++++++++++++++++++++++++
              for d in range(n_dense_layers):
                dense_layer = keras.layers.Dense(
                        units=dense_architecture[d],
                        kernel_initializer=dense_kernel_inits[d],
                        bias_initializer=dense_bias_inits[d],
                        name=encoder_name+"_dense"+str(d),
                        dtype=tf.keras.mixed_precision.Policy(self.precision)
                    )
                dense_activation = self.get_activation_function(dense_activations[d],name=encoder_name+"_dense_act"+str(d))

                x_enc = dense_layer(x_enc)
                x_enc = dense_activation(x_enc)
              #++++++++++++++++++++++++++
        
        # Check if the latent layer is two dimenstional:
        if latent_is_2d == True:
           # Handle the encoding / latent layer:        
           encoding_layer = keras.layers.Dense(
                units=latent_dim*latent_dim,
                activation=latent_activation,
                name=encoder_name+"_latent_layer",
                kernel_initializer=latent_kernel_init,
                bias_initializer="zeros",
                dtype=tf.keras.mixed_precision.Policy(self.precision)
           )
           encoder_outputs = encoding_layer(x_enc)
           encoder_outputs = keras.layers.Reshape(target_shape=(latent_dim,latent_dim))(encoder_outputs)
           return keras.models.Model(encoder_inputs,encoder_outputs,name=encoder_name)
        
        else: # Or not...

           # Handle the encoding / latent layer:        
           encoding_layer = keras.layers.Dense(
                units=latent_dim,
                activation=latent_activation,
                name=encoder_name+"_latent_layer",
                kernel_initializer=latent_kernel_init,
                bias_initializer="zeros",
                dtype=tf.keras.mixed_precision.Policy(self.precision)
           )
           encoder_outputs = encoding_layer(x_enc)
           return keras.models.Model(encoder_inputs,encoder_outputs,name=encoder_name)
    #*************************

    # Decoder:    
    #*************************
    def get_decoder(self,latent_dim,latent_is_2d,reshape_dimensions,conv_architecture,conv_activations,conv_kernel_inits,conv_bias_inits,kernel_sizes,strides,dense_architecture,dense_activations,dense_kernel_inits,dense_bias_inits,output_filter,output_kernel_size,output_strides,output_activation,max_pooling,decoder_name):
        # Get the number of conv. / dense layers:
        n_conv_layers = len(conv_architecture)
        n_dense_layers = len(dense_architecture)

        # Define the decoder inputs:
        decoder_inputs = None
        z_dec = None
        
        # First, check if the latent input is 2D matrix:
        if latent_is_2d == True:
           decoder_inputs = keras.layers.Input(
             shape=(latent_dim,latent_dim,),
             name=decoder_name+"_input"
           )
           z_dec = keras.layers.Flatten()(decoder_inputs)
        else: 
          decoder_inputs = keras.layers.Input(
            shape=(latent_dim,),
            name=decoder_name+"_input"
          )
          z_dec = decoder_inputs

        # Add a dense part, if requested:
        if n_dense_layers > 0:
            #++++++++++++++++++++++++++
            for d in range(n_dense_layers):
                dense_layer = keras.layers.Dense(
                        units=dense_architecture[d],
                        kernel_initializer=dense_kernel_inits[d],
                        bias_initializer=dense_bias_inits[d],
                        name=decoder_name+"_dense"+str(d),
                        dtype=tf.keras.mixed_precision.Policy(self.precision)
                )
                dense_activation = self.get_activation_function(dense_activations[d],name=decoder_name+"_dense_act"+str(d))

                z_dec = dense_layer(z_dec)
                z_dec = dense_activation(z_dec)
            #++++++++++++++++++++++++++
                
        # Now we need to reshape in order to translate everything back 
        # from the 1D latent space to the conv. space
        reshaping_layer = keras.layers.Dense(
                units=reshape_dimensions[0]*reshape_dimensions[1]*reshape_dimensions[2],
                activation='relu',
                kernel_initializer='he_normal',
                bias_initializer='zeros',
                name=decoder_name+'_conv_reshape',
                dtype=tf.keras.mixed_precision.Policy(self.precision)
        )
        z_dec = reshaping_layer(z_dec)
        # Now convert everything to 2D:
        z_dec = keras.layers.Reshape(target_shape=reshape_dimensions)(z_dec)

        # Translate everything back via transpose conv.:
        #++++++++++++++++++++++++++
        for k in range(n_conv_layers):
            transpose_conv_layer = keras.layers.Conv2DTranspose(
                filters=conv_architecture[k], 
                kernel_size=kernel_sizes[k], 
                strides=strides[k], 
                padding='same',
                kernel_initializer=conv_kernel_inits[k],
                bias_initializer=conv_bias_inits[k],
                name=decoder_name+"_convT"+str(k),
                dtype=tf.keras.mixed_precision.Policy(self.precision)
            )
            transpose_conv_activation = self.get_activation_function(conv_activations[k],name=decoder_name+"_convT_act"+str(k))

            z_dec = transpose_conv_layer(z_dec)
            z_dec = transpose_conv_activation(z_dec)
        #++++++++++++++++++++++++++
            
        # Add an output layer:
        output_layer = keras.layers.Conv2DTranspose(
                filters=output_filter,
                kernel_size=output_kernel_size,
                strides=output_strides,
                activation=output_activation,
                padding='same',
                name=decoder_name+"_output",
                dtype=tf.keras.mixed_precision.Policy(self.precision)
        )
        x_rec = output_layer(z_dec)
        
        # Undo the max. pooling, if existent:
        if max_pooling is not None and max_pooling[0] > 0 and max_pooling[1] > 0:
            x_rec = keras.layers.UpSampling2D(
               size=(max_pooling[0],max_pooling[1]),
               interpolation='nearest',
               dtype=tf.keras.mixed_precision.Policy(self.precision)
            )(x_rec)

        return keras.models.Model(decoder_inputs,x_rec,name=decoder_name)
    #*************************



    # Experimental: Define encoder with conditional input --> So that we can use it for a diffusion model:

    # Encoder for a diffusion model:
    #*************************
    def get_encoder_for_diffusion(self,input_dimensions,input_dimensions2,input2_processing_fn,conv_architecture,conv_activations,conv_kernel_inits,conv_bias_inits,kernel_sizes,strides,dense_architecture,dense_activations,dense_kernel_inits,dense_bias_inits,latent_dim,latent_activation,latent_kernel_init,latent_is_2d,max_pooling,encoder_name):
        # Get the number of conv. / dense layers:
        n_conv_layers = len(conv_architecture)
        n_dense_layers = len(dense_architecture)
        
        # Define the encoder input:
        encoder_inputs = keras.layers.Input(
            shape=input_dimensions,
            name=encoder_name+"_input"
        )

        encoder_inputs_2 = keras.layers.Input(
            shape=input_dimensions2,
            name=encoder_name+"_input_2"
        )

        # Apply max-pooling, if requested:
        x_enc = encoder_inputs
        if max_pooling is not None and max_pooling[0] > 0 and max_pooling[1] > 0:
            x_enc = keras.layers.MaxPooling2D(
                pool_size=(max_pooling[0],max_pooling[1]),
                dtype=tf.keras.mixed_precision.Policy(self.precision)
            )(x_enc)

        x_enc_2 = input2_processing_fn(x_enc,encoder_inputs_2)
        x_enc = keras.layers.Concatenate()([x_enc,x_enc_2])

        # Add conv. parts:
        #++++++++++++++++++++++++++
        for k in range(n_conv_layers):
            conv_laver =  keras.layers.Conv2D(
                    filters=conv_architecture[k], 
                    kernel_size=kernel_sizes[k], 
                    strides=strides[k],
                    kernel_initializer=conv_kernel_inits[k],
                    bias_initializer=conv_bias_inits[k],
                    name=encoder_name+"_conv"+str(k),
                    dtype=tf.keras.mixed_precision.Policy(self.precision) 
            )
            conv_activation = self.get_activation_function(conv_activations[k],name=encoder_name+"_conv_act"+str(k))

            x_enc = conv_laver(x_enc)
            x_enc = conv_activation(x_enc)
        #++++++++++++++++++++++++++
        
        # Add a flattening layer:
        x_enc = keras.layers.Flatten(dtype=tf.keras.mixed_precision.Policy(self.precision))(x_enc)
        
        # Add a dense part, if wanted:
        if n_dense_layers > 0:
              #++++++++++++++++++++++++++
              for d in range(n_dense_layers):
                dense_layer = keras.layers.Dense(
                        units=dense_architecture[d],
                        kernel_initializer=dense_kernel_inits[d],
                        bias_initializer=dense_bias_inits[d],
                        name=encoder_name+"_dense"+str(d),
                        dtype=tf.keras.mixed_precision.Policy(self.precision)
                    )
                dense_activation = self.get_activation_function(dense_activations[d],name=encoder_name+"_dense_act"+str(d))

                x_enc = dense_layer(x_enc)
                x_enc = dense_activation(x_enc)
              #++++++++++++++++++++++++++
        
        # Check if the latent layer is two dimenstional:
        if latent_is_2d == True:
           # Handle the encoding / latent layer:        
           encoding_layer = keras.layers.Dense(
                units=latent_dim*latent_dim,
                activation=latent_activation,
                name=encoder_name+"_latent_layer",
                kernel_initializer=latent_kernel_init,
                bias_initializer="zeros",
                dtype=tf.keras.mixed_precision.Policy(self.precision)
           )
           encoder_outputs = encoding_layer(x_enc)
           encoder_outputs = keras.layers.Reshape(target_shape=(latent_dim,latent_dim))(encoder_outputs)
           return keras.models.Model(encoder_inputs,encoder_outputs,name=encoder_name)
        
        else: # Or not...

           # Handle the encoding / latent layer:        
           encoding_layer = keras.layers.Dense(
                units=latent_dim,
                activation=latent_activation,
                name=encoder_name+"_latent_layer",
                kernel_initializer=latent_kernel_init,
                bias_initializer="zeros",
                dtype=tf.keras.mixed_precision.Policy(self.precision)
           )
           encoder_outputs = encoding_layer(x_enc)
           return keras.models.Model(inputs=[encoder_inputs,encoder_inputs_2],outputs=encoder_outputs,name=encoder_name)
    #*************************


