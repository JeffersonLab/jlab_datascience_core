from jlab_datascience_toolkit.core.jdst_analysis import JDSTAnalysis
import matplotlib.pyplot as plt
import os
import yaml
import inspect
import logging

class LearningCurveVisualizer(JDSTAnalysis):
    '''
    Simple class to visualize the learning curves produced during model training.

    Input(s):
    i) Dictionary with all loss curves

    Output(s):
    i) .png files visualizing the learning curves 
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "learning_curve_visualizer"
        
        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)

        # Get plots that shall be produced:
        self.plots = self.config['plots']
        # Get the corresponding plot labels, plot legends and the names of each individual plot:
        self.plot_labels = self.config['plot_labels']
        self.plot_legends = self.config['plot_legends']
        self.plot_names = self.config['plot_names']

        # Cosmetics:
        self.fig_size = self.config['fig_size']
        self.line_width = self.config['line_width']
        self.font_size = self.config['font_size']
        self.leg_font_size = self.config['leg_font_size']

        # Set font size:
        plt.rcParams.update({'font.size':self.font_size})
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Get the output location and create proper folders:
        self.output_loc = self.config['output_loc']
        self.plot_loc = self.output_loc+"/learning_curves"

        os.makedirs(self.output_loc,exist_ok=True)
        os.makedirs(self.plot_loc,exist_ok=True)
    #*********************************************

    # Check the data type:
    #*********************************************
    def check_input_data_type(self,data):
        if isinstance(data,dict) == True:
            if bool(dict) == False:
                logging.error(f">>> {self.module_name}: Your dictionary {data} is empty. Please check. Going to return None. <<<")
                return False
            return True
        
        else:
            logging.error(f">>> {self.module_name}: The data type you provided {type(data)} is not a dictionary. Please check. Going to return None. <<<")
            return False
    #*********************************************

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

    # Run the entire analysis:
    #*********************************************
    # Peoduce a single plot, based on scores and legends:
    def produce_single_plot(self,history,scores,legend_entries,axis):
        if legend_entries is None:
            #++++++++++++++++
            for s in scores:
                if s in history:
                  metric = history[s]
                  x = [k for k in range(1,1+len(metric))]
                  axis.plot(x,metric,linewidth=self.line_width)
            #++++++++++++++++
        
        else:
            #++++++++++++++++
            for s,l in zip(scores,legend_entries):
                if s in history:
                  metric = history[s]
                  x = [k for k in range(1,1+len(metric))]
                  axis.plot(x,metric,linewidth=self.line_width,label=l)
            #++++++++++++++++
            axis.legend(fontsize=self.leg_font_size)


    def run(self,training_history):
        if self.check_input_data_type(training_history):
          # Loop through all plots that we wish to produce:
          #+++++++++++++++++++++++
          for plot in self.plots:
            # Create a canvas to draw on:
            fig,ax = plt.subplots(figsize=self.fig_size)
            
            scores = self.plots[plot]
            
            legend_entries = self.plot_legends.get(plot,None)
            labels = self.plot_labels.get(plot,None)
            name = self.plot_names.get(plot,None)

            if legend_entries is not None:
                assert len(legend_entries) == len(scores), logging.error(f">>> {self.module_name}: Number of legend entries {legend_entries} does not match the number of available score {scores} <<<")
                
            # Produce a nice plot:
            self.produce_single_plot(training_history,scores,legend_entries,ax)
            ax.grid(True)    
            
            if labels is not None:
                   assert len(labels) == 2, logging.error(f">>> {self.module_name}: Number of plot labels {labels} does not match exptected number of two entries <<<")

                   # Add labels if available:
                   ax.set_xlabel(labels[0])
                   ax.set_ylabel(labels[1])

            # Store the figure somewhere:
            if name is not None:
                 fig.savefig(self.plot_loc+"/"+name+".png")
                 plt.close(fig)
          #+++++++++++++++++++++++

        else:
            return None
    #*********************************************

    # Save and load are not active here:
    #*********************************************
    def save(self):
        pass

    def load(self):
        pass
    #*********************************************



