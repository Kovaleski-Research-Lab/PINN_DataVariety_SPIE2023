#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import signal
import logging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core import datamodule, lrn, custom_logger
from utils import parameter_manager, model_loader

#--------------------------------
# Initialize: Training
#--------------------------------
  
def run(params):
    #logging.basicConfig(level=logging.DEBUG)

    # Initialize: Parameter manager
    pm = parameter_manager.Parameter_Manager(params=params)
          
    # Initialize: Seeding
    if(pm.seed_flag):
        seed_everything(pm.seed_value, workers = True)

    # Initialize: The model
    model = model_loader.select_model(pm)

    # Initialize: The datamodule
    data = datamodule.select_data(pm.params_datamodule)

    # Initialize: The logger
    logger = custom_logger.Logger(all_paths=pm.all_paths, name=pm.model_name, version=0)

    # Initialize:  PytorchLighting model checkpoint
    checkpoint_path = os.path.join(pm.path_root, pm.path_model_lrn)
    checkpoint_callback = ModelCheckpoint(dirpath = checkpoint_path)
    logging.debug(f'Checkpoint path: {checkpoint_path}')

    logging.debug('Setting matmul precision to HIGH')
    torch.set_float32_matmul_precision('high')

    # Initialize: PytorchLightning Trainer
    if(pm.gpu_flag and torch.cuda.is_available()):
        logging.debug("Training with GPUs")
        trainer = Trainer(logger = logger, accelerator = "cuda", num_nodes = 1, 
                          check_val_every_n_epoch = pm.valid_rate, num_sanity_val_steps = 1,
                          devices = pm.gpu_list, max_epochs = pm.num_epochs, 
                          deterministic=True, enable_progress_bar=True, enable_model_summary=True,
                          default_root_dir = pm.path_root, callbacks = [checkpoint_callback],
                          plugins = [SLURMEnvironment(requeue_signal=signal.SIGHUP),],
                          )
    else:
        logging.debug("Training with CPUs")
        trainer = Trainer(logger = logger, accelerator = "cpu", max_epochs = pm.num_epochs, 
                          num_sanity_val_steps = 0, default_root_dir = pm.path_results, 
                          check_val_every_n_epoch = pm.valid_rate, callbacks = [checkpoint_callback])

    # Train
    trainer.fit(model,data)

    # Dump config
    yaml.dump(params, open(os.path.join(pm.path_root, f'{pm.results_path}/params.yaml'), 'w'))
    #torch.save(model.state_dict(), 'test_state_dict.dat')
