#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import os
import csv
import shutil
import pickle
import logging
import numpy as np

from PIL import Image

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.logger import rank_zero_experiment

#--------------------------------
# Initialize: Experiment Logger
#--------------------------------

class Writer:
    def __init__(self, path, name = "loss.csv"):
        
        logging.debug("custom_logger.py - Initializing Writer")

        self.name = name
        self.path = path
        self.metrics = []

        # Initialize: Paths 

        self.path_metrics = os.path.join(self.path, self.name)
        self.path_train = os.path.join(self.path, "train_info")
        self.path_valid = os.path.join(self.path, "valid_info")

        if(os.path.exists(self.path_valid)):
            logging.debug("Writer | Removing {}".format(self.path_valid))
            shutil.rmtree(self.path_valid)
        if(os.path.exists(self.path_train)):
            logging.debug("Writer | Removing {}".format(self.path_train))
            shutil.rmtree(self.path_train)


        os.makedirs(self.path_valid)
        os.makedirs(self.path_train)

    #----------------------------
    # Update: Performance Metrics 
    #----------------------------

    def log_metrics(self, metrics_dict, step = None):
        logging.debug("Writer | logging metric dictionary")
        if step is None:
            step = len(self.metrics)
        self.metrics.append(metrics_dict)

    #----------------------------
    # Logging: Results
    #----------------------------

    def log_results(self, results, epoch, mode, count = 5):
        logging.debug("Writer | logging and saving results")

        name = str(epoch).zfill(count) + ".pkl"

        if(mode == "train"):
            path = self.path_train
        else:
            path = self.path_valid

        path_save = os.path.join(path, name)
        pickle.dump(results, open(path_save, "wb"))

    #----------------------------
    # Saving: Metrics
    #----------------------------

    def save(self):
        logging.debug("Writer | saving metric dictionary")

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with open(self.path_metrics, "w", newline="") as f:
            self.writer = csv.DictWriter(f, fieldnames=metrics_keys)
            self.writer.writeheader()
            self.writer.writerows(self.metrics)

#--------------------------------
# Initialize: Experiment Logger
#--------------------------------

class Logger(Logger):
    def __init__(self, all_paths, name = "default", version = None, prefix = ""):
        super().__init__()
        logging.debug("custom_logger.py - Initializing Logger")

        self._name = name 
        self._prefix = prefix
        self._experiment = None
        self._version = version
        root = all_paths['path_root']
        if self.name == 'LRN':
            self._save_dir = os.path.join(root,all_paths['path_results_lrn'])
        elif self.name == 'classifier':
            self._save_dir = os.path.join(root,all_paths['path_results_classifier'])
        elif self.name == 'cooperative':
            self._save_dir = os.path.join(root,all_paths['path_results_cooperative'])
        self.display_paths(all_paths)
        
    @rank_zero_only
    def display_paths(self, all_paths):
      
        logging.debug("\nLogger | Experiment Paths:")
        for current_key in all_paths.keys():
            logging.debug("Logger | Path %s: %s" % (current_key, all_paths[current_key]))

    #----------------------------
    # Gather: Path (Root Folder)
    #----------------------------
    
    @property
    def root_dir(self) -> str:
        #if not self.name:
        #    return self.save_dir

        #return os.path.join(self.save_dir, self.name)
        return self.save_dir

    #----------------------------
    # Gather: Path (Log Folder)
    #----------------------------

    @property
    def log_dir(self):

        #version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        #log_dir = os.path.join(self.root_dir, version)

        return self.root_dir

    #----------------------------
    # Gather: Path (Save Folder)
    #----------------------------

    @property
    def save_dir(self):

        return self._save_dir

    #----------------------------
    # Gather: Experiment Version
    #----------------------------

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    #----------------------------
    # Gather: Experiment Title
    #----------------------------

    @property
    def name(self):
        return self._name

    #----------------------------
    # Log: Performance Metrics
    #----------------------------

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.experiment.log_metrics(metrics, step)

    #----------------------------
    # Log: Model Hyperparameters
    #----------------------------

    @rank_zero_only
    def log_hyperparams(self, params):

        pass

    #----------------------------
    # Initialize: Saver Actions
    #----------------------------

    @rank_zero_only
    def save(self):

        super().save()
        self.experiment.save()

    #----------------------------
    # Run: Post Training Code
    #----------------------------

    @rank_zero_only
    def finalize(self, status):
        self.save()

    #----------------------------
    # Run: Logger-Writer Object
    #----------------------------

    @property
    @rank_zero_experiment
    def experiment(self):

        if(self._experiment):

            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)

        self._experiment = Writer(path = self.log_dir)

        return self._experiment

