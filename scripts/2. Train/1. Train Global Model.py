# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

# Add project root to path to find constants.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from constants import PROJECT_ROOT_DIR, BASE_MODEL_FPATH

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment
checkpoint = BASE_MODEL_FPATH
root_dir = PROJECT_ROOT_DIR

# %%
config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoser",
                project_root_dir=root_dir, joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0") 

# %%
# instantiate model and data (load weights if fine-tuning from checkpoint)
model = IMUPoserModel.load_from_checkpoint(
    checkpoint,
    config=config
)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path 

# %%
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=5)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

# fast_dev_run = fast_dev_run 
trainer = pl.Trainer(logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)

# %%
trainer.fit(model, datamodule=datamodule)

# %%
with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
