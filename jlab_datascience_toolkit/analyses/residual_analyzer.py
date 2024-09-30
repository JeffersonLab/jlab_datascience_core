from jlab_datascience_toolkit.core.jdst_analysis import JDSTAnalysis
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect
import yaml
import imageio
import logging

class ResidualAnalyzer(JDSTAnalysis):
    '''
    Simple class to compare the input and reconstructed (e.g. from an autoencoder) data by computing residuals.
    '''

    # Initialize:
    #****************************
    def __init__(self,path_to_cfg,user_config={}):
        # Define the module and module name:
        self.module_name = "residual_analyzer"

         # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])
        
        # General settings:
        self.output_loc = self.config['output_loc']
        self.residual_dir = self.output_loc+"/residuals"
        self.real_data_name = self.config['real_data_name']
        self.rec_data_name = self.config['rec_data_name']
    
        # Define the reduction mode, i.e. how the residuals are computed:
        self.reduction_mode = self.config["reduction_mode"]
        self.reduction_axis = self.config["reduction_axis"]

        # Settings to plot images:
        self.imageplot_figsize = self.config["imageplot_figsize"]
        self.imageplot_noaxes = self.config["imageplot_noaxes"]

        # Settings to store all images as movies:
        self.movie_duration = self.config["movie_duration"]
        
        os.makedirs(self.output_loc,exist_ok=True)
        os.makedirs(self.residual_dir,exist_ok=True)
    #****************************

    # Check input data type:
    #****************************
    def check_input_data_type(self,data):
        if isinstance(data,dict) == True:
            if bool(dict) == False:
                logging.error(f">>> {self.module_name}: Your dictionary {data} is empty. Please check. Going to return None. <<<")
                return False
            
            return True
        
        else:
            logging.error(f">>> {self.module_name}: The data type you provided {type(data)} is not a dictionary. Please check. Going to return None. <<<")
            return False
    #****************************

    # Provide information about this module:
    #*********************************************
    def get_info(self):
        print(inspect.getdoc(self))
    #*********************************************

    # Handle configurations:
    #*********************************************
    # Load the config:
    def load_config(self,path_to_cfg,user_config):
        with open(path_to_cfg, 'r') as file:
            cfg = yaml.safe_load(file)
        
        # Overwrite config with user settings, if provided
        try:
            if bool(user_config):
              #++++++++++++++++++++++++
              for key in user_config:
                cfg[key] = user_config[key]
              #++++++++++++++++++++++++
        except:
            logging.exception(">>> " + self.module_name +": Invalid user config. Please make sure that a dictionary is provided <<<") 

        return cfg
    
    #-----------------------------

    # Store the config:
    def save_config(self,path_to_config):
        with open(path_to_config, 'w') as file:
           yaml.dump(self.config, file)
    #*********************************************
    
    # Compute the residuals:
    #****************************
    # Make sure that we are operating in 3 dimensions:
    def image_dimension_check(self,image):
        if len(image.shape) < 4:
            return np.expand_dims(image,3)
        return image
    
    #-----------------------

    # Now compute the residuals:
    def compute_residuals(self,x_real,x_rec):
        residual = self.image_dimension_check(x_real) - self.image_dimension_check(x_rec)

        if self.reduction_mode.lower() == "mean":
            return np.mean(residual,axis=self.reduction_axis)
        elif self.reduction_mode.lower() == "abs_mean":
            return np.mean(np.abs(residual),axis=self.reduction_axis)
        elif self.reduction_mode.lower() == "squared_mean":
            return np.mean(np.square(residual),axis=self.reduction_axis)
        else:
            logging.warning(f">>> {self.module_name}: Reduction mode {self.reduction_mode} does not exist. Going to use mean reduction mode (default)<<<")
            return np.mean(residual,axis=self.reduction_axis)
    #****************************

    # Generic function to plot an image:
    #****************************
    def plot_images(self,real_images,rec_images,residual_images,path,name):

        #++++++++++++++++++++++
        for i in range(real_images.shape[0]):
            fig,ax = plt.subplots(1,3,figsize=self.imageplot_figsize)
            
            ax[0].set_title('Original')
            ax[0].imshow(real_images[i])

            if self.imageplot_noaxes:
                ax[0].set_axis_off()

            ax[1].set_title('Reconstructed')
            ax[1].imshow(rec_images[i])

            if self.imageplot_noaxes:
                ax[1].set_axis_off()

            ax[2].set_title('Residual')
            ax[2].imshow(residual_images[i])

            if self.imageplot_noaxes:
                ax[2].set_axis_off()

            fig.savefig(path+"/"+name+"_"+str(i)+".png")
            plt.close(fig)
        #++++++++++++++++++++++
    #****************************

    # Translate images to a single movie (just a little gimmick to better visualize the data)
    #****************************
    def png_to_movie(self,png_path,movie_path,movie_name):
        filenames = []
        #++++++++++++++++++++++++++
        for file in os.listdir(png_path):
           if ".png" in file:
             filenames.append(os.path.join(png_path, file))
        #++++++++++++++++++++++++++

        filenames = sorted(filenames)
        images = []
        shape = None
        for filename in filenames:
          img = imageio.imread(filename)
          if shape == None:
              shape = img.shape
           
          images.append(img)
        
        imageio.mimsave(os.path.join(movie_path,movie_name+'.gif'),images, duration=self.movie_duration)
    #****************************

    # Put it all together:
    #****************************
    def run(self,data_dict):
        if self.check_input_data_type(data_dict):
           x_real = data_dict[self.real_data_name]
           x_rec = data_dict[self.rec_data_name]

           # Compute the residuals first:
           residuals = self.compute_residuals(x_real,x_rec)

           # Plot the residuals
           self.plot_images(x_real,x_rec,residuals,self.residual_dir,"res")

           # Store everything as a movie, if duration is specified:
           if self.movie_duration > 0.0:
              self.png_to_movie(self.residual_dir,self.residual_dir,"res_mov")
        else:
            return None
    #****************************
    
    # Save and load are not active here:
    #****************************
    def save(self):
        pass

    def load(self):
        pass
    #****************************

