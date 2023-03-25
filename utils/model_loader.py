#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core import datamodule, lrn, custom_logger



def select_model(pm):
    logging.debug("select_model.py - Selecting model") 
    model = None
    if pm.lrn:
        logging.debug("Select_Model | selecting LRN")
        assert model is None
        model = lrn.LRN(pm.params_model_lrn, pm.params_propagator, pm.params_modulator)
        if pm.load_checkpoint_lrn:
            model.load_from_checkpoint(pm.path_checkpoint_lrn,
                                           params = (pm.params_model_lrn, pm.params_propagator, pm.params_modulator),
                                           strict = True)
    
    else:
        logging.error("model_loader.py | Failed to select model")
        exit()
    assert model is not None

    return model
