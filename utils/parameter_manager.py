import logging
import torch
import yaml
import traceback
import sys
import os
class Parameter_Manager():
    def __init__(self, config = None,  params = None):
        logging.debug("parameter_manager.py - Initializing Parameter_Manager")

        if config is not None:
            self.open_config(config)
        if params is not None:
            self.params = params

        self.parse_params(self.params)

    def open_config(self, config_file):
        try:
            with open(config_file) as c:
                self.params = yaml.load(c, Loader = yaml.FullLoader)
        except Exception as e:
            logging.error(e)
            sys.exit()
            
    def parse_params(self, params):
        try:
            # Load: Paths 
            self.path_root = params['path_root']
            self.path_data = params['path_data']
            self.path_model = params['path_model']
            self.path_train = params['path_train']
            self.path_valid = params['path_valid']
            self.path_results = params['path_results']
            self._path_checkpoint_lrn = params['path_checkpoint_lrn']
 
            # Load: Trainer Params
            self.batch_size = params['batch_size']
            self.num_epochs = params['num_epochs']
            self.valid_rate = params['valid_rate']
            self.accelerator = params['accelerator']
            self.gpu_flag, self.gpu_list = params['gpu_config']

            # Load: LRN Model Params
            self.optimizer_lrn = params['optimizer_lrn']
            self.num_layers_lrn = params['num_layers_lrn']
            self.learning_rate_lrn = params['learning_rate_lrn']
            self.transfer_learn_lrn = params['transfer_learn_lrn']
            self.load_checkpoint_lrn = params['load_checkpoint_lrn']
            self.objective_function_lrn = params['objective_function_lrn']

            # Load: Datamodule Params
            self._which = params['which']
            self.n_cpus = params['n_cpus']
            self._data_split = params['data_split']
            self.transforms = params['wavefront_transform']
            
            # Load: Physical Params
            self._distance = params['distance']
            if(not(isinstance(self._distance, torch.Tensor))):
                self._distance = torch.tensor(self._distance)
            self._wavelength = torch.tensor(float(params['wavelength']))
            # Propagator
            self.Nxp = params['Nxp']
            self.Nyp = params['Nyp']
            self.Lxp = params['Lxp']
            self.Lyp = params['Lyp']
            self._adaptive = params['adaptive']
            #Modulator
            self.Nxm = params['Nxm']
            self.Nym = params['Nym']
            self.Lxm = params['Lxm']
            self.Lym = params['Lym']
            self.modulator_type = params['modulator_type'] 
            self._phase_initialization = params['phase_initialization']
            self.amplitude_initialization = params['amplitude_initialization']


            # Determine the type of experiment we are running
            self.lrn = params['LRN']
            self.model_id = params['model_id']

            try:
                self.jobid = os.environ['SLURM_JOB_ID']
            except:
                self.jobid = 0

            # Only LRN
            self.model_name = 'LRN'
            self.path_model_classifier = None
            self.path_results_classifier = None
            self.path_model_cooperative = None
            self.path_results_cooperative = None
            self.path_model_lrn = f"{self.path_model}/{self.model_name}/model_{self.model_id}/"
            self.path_results_lrn = f"{self.path_results}/{self.model_name}/model_{self.model_id}/"
            self.results_path = self.path_results_lrn

            self.seed_flag, self.seed_value = params['seed']
        
            self.collect_params()


        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            sys.exit()

    def collect_params(self):
        logging.debug("Parameter_Manager | collecting parameters")
        self._params_model_lrn = {
                                'optimizer'             : self.optimizer_lrn,
                                'num_layers'            : self.num_layers_lrn, 
                                'learning_rate'         : self.learning_rate_lrn,
                                'transfer_learn'        : self.transfer_learn_lrn, 
                                'objective_function'    : self.objective_function_lrn,
                                'load_checkpoint_lrn'   : self.load_checkpoint_lrn,
                                }

             
        self._params_propagator = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'Lxp'           : self.Lxp, 
                                'Lyp'           : self.Lyp,
                                'distance'      : self._distance,
                                'adaptive'      : self.adaptive,
                                'batch_size'    : self.batch_size,
                                'wavelength'    : self._wavelength, 
                                }
                
        self._params_modulator = {
                                'Nxm'                       : self.Nxm, 
                                'Nym'                       : self.Nym, 
                                'Lxm'                       : self.Lxm, 
                                'Lym'                       : self.Lym,
                                'wavelength'                : self._wavelength, 
                                'modulator_type'            : self.modulator_type,
                                'phase_initialization'      : self._phase_initialization,
                                'amplitude_initialization'  : self.amplitude_initialization,
                                }

        self._params_datamodule = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'which'         : self._which,
                                'n_cpus'        : self.n_cpus,
                                'path_root'     : self.path_root, 
                                'path_data'     : self.path_data, 
                                'batch_size'    : self.batch_size, 
                                'data_split'    : self.data_split,
                                'transforms'    : self.transforms,
                                }

        self._params_trainer = {
                            'num_epochs'    : self.num_epochs, 
                            'valid_rate'    : self.valid_rate,
                            'accelerator'   : self.accelerator, 
                            }
 

        self._all_paths = {
                        'path_root'                     : self.path_root, 
                        'path_data'                     : self.path_data, 
                        'path_model'                    : self.path_model,
                        'path_train'                    : self.path_train, 
                        'path_valid'                    : self.path_valid,
                        'path_results'                  : self.path_results, 
                        'path_model_lrn'                : self.path_model_lrn, 
                        'path_results_lrn'              : self.path_results_lrn, 
                        'path_checkpoint_lrn'           : self._path_checkpoint_lrn,
                        }

    @property 
    def params_model_lrn(self):         
        return self._params_model_lrn


    @property
    def params_propagator(self):
        return self._params_propagator                         

    @property
    def params_modulator(self):
        return self._params_modulator

    @property
    def params_datamodule(self):
        return self._params_datamodule

    @property 
    def params_trainer(self):
        return self._params_trainer

    @property
    def all_paths(self):
        return self._all_paths 

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        logging.debug("Parameter_Manager | setting distance to {}".format(value))
        self._distance = value
        self.collect_params()
        
    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        logging.debug("Parameter_Manager | setting wavelength to {}".format(value))
        self._wavelength = value
        self.collect_params()
    
    @property
    def path_checkpoint_lrn(self):
        return self._path_checkpoint_lrn

    @path_checkpoint_lrn.setter
    def path_checkpoint_lrn(self, value):
        logging.debug("Parameter_Manager | setting path_checkpoing_lrn to {}".format(value))
        self._path_checkpoint_lrn = value
        self.collect_params()

    @property
    def which(self):
        return self._which

    @which.setter
    def which(self, value):
        logging.debug("Parameter_Manager | setting which to {}".format(value))
        self._which = value
        self.collect_params()

    @property
    def adaptive(self):
        return self._adaptive

    @adaptive.setter
    def adaptive(self, value):
        logging.debug("Parameter_Manager | setting adaptive to {}".format(value))
        self._adaptive = value
        self.collect_params()

    @property
    def phase_initialization(self):
        return self._phase_initialization

    @phase_initialization.setter
    def phase_initialization(self, value):
        logging.debug("Parameter_Manager | setting phase_initialization to {}".format(value))
        self._phase_initialization = value
        self.collect_params()

    @property
    def data_split(self):
        return self._data_split

    @data_split.setter
    def data_split(self, value):
        logging.debug("Parameter_Manager | setting data_split to {}".format(value))
        self._data_split = value
        self.collect_params()

if __name__ == "__main__":
    import yaml
    params = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    pm = Parameter_Manager(params = params)
    print(pm.path_model)

