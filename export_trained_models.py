import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import importlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import train_utils
import rvsr
import padding
from job import presets, get_preset_hpars, hpars
from paths_config import paths_config

step = hpars["checkpoint_steps"][-1]

os.makedirs(paths_config["trained_models_folder"], exist_ok=True)

def find_preset_seed_run_folders():
    preset_seed_run_folders = {}
    for preset in presets.keys():
        preset_hpars = get_preset_hpars(preset)
        preset_seed_run_folders[preset] = []
        i = 0
        while True:
            candidate_folders = [os.path.join(paths_config["train_runs_folder"], f"{preset}_s{i}")]
            if "compat_preset" in preset_hpars:
                candidate_folders.append(os.path.join(paths_config["train_runs_folder"], f"{preset_hpars["compat_preset"]}_s{i}"))
            for candidate_folder in candidate_folders:
                if os.path.exists(candidate_folder):
                    preset_seed_run_folders[preset].append(candidate_folder)
                    break
            if len(preset_seed_run_folders[preset]) < i:
                # Folder for seed i not found, assume that subsequent seeds won't be found
                break
            i += 1
    return preset_seed_run_folders

def enter_compat_mode():
    jax._src.cache_key.custom_hook = lambda: "compat"
    sys.modules['dataloading'] = compat_dataloading
    sys.modules['train_utils'] = compat_train_utils
    sys.modules['rvsr'] = compat_rvsr
    sys.modules['padding'] = compat_padding
    importlib.reload(sys.modules['train_utils'])
    importlib.reload(sys.modules['rvsr'])
    importlib.reload(sys.modules['padding'])

def exit_compat_mode():
    jax._src.cache_key.custom_hook = lambda: "non-compat"
    sys.modules['train_utils'] = train_utils
    sys.modules['padding'] = padding
    sys.modules['rvsr'] = rvsr
    importlib.reload(sys.modules['train_utils'])
    importlib.reload(sys.modules['rvsr'])
    importlib.reload(sys.modules['padding'])

if __name__ == "__main__":
    preset_to_seed_run_folders = find_preset_seed_run_folders()
    for preset, folder_list in preset_to_seed_run_folders.items():
        for seed, folder in enumerate(folder_list):
            print(f"Preset: {preset}, seed: {seed}, run folder: {folder}")
            chkpoint_folder = f"{folder}/{str(step).zfill(8)}"
            if os.path.exists(chkpoint_folder):
                try:
                    trainer = train_utils.SuperresolutionTrainer(folder=folder, resume_checkpoint=step)
                except (ModuleNotFoundError, AttributeError):
                    print("Checkpoint uses older module versions, reloading with compat modules")
                    from compat import train_utils as compat_train_utils
                    from compat import dataloading as compat_dataloading
                    from compat import padding as compat_padding
                    from compat import rvsr as compat_rvsr
                    enter_compat_mode()
                    trainer = compat_train_utils.SuperresolutionTrainer(folder=folder, resume_checkpoint=step)
                    exit_compat_mode()
                model_eqx_filename = os.path.join(paths_config["trained_models_folder"], f"{preset}_s{seed}.eqx")
                print(f"Saving model to {model_eqx_filename}")
                rvsr.save_rvsr_weights(trainer.model, model_eqx_filename)
                train_loss_steps = np.array(trainer.loss_steps, dtype=np.int32)
                train_losses = np.array(trainer.losses, dtype=np.float16)
                test_losses = np.array(trainer.test_losses, dtype=np.float16)
                test_loss_steps = np.array(trainer.test_loss_steps, dtype=np.int32)
            else:
                print(f"No checkpoint {step}. Loading loss histories")
                try:
                    train_losses = np.loadtxt(f"{folder}/train_history.csv", delimiter=",").T
                    train_loss_steps = train_losses[0].astype(np.int32)
                    train_losses = train_losses[1].astype(np.float16)
                except:
                    print("No train loss history found")
                    train_losses = None
                try:
                    test_losses = np.loadtxt(f"{folder}/test_history.csv", delimiter=",").T
                    test_loss_steps = test_losses[0].astype(np.int32)
                    test_losses = test_losses[1].astype(np.float16)
                except:
                    print("No train loss history found")
                    test_losses = None
            if (train_losses is not None) or (test_losses is not None):
                print(f"Saving loss histories to {os.path.join(paths_config["trained_models_folder"], f"{preset}_s{seed}*")}")
            if train_losses is not None:
                train_loss_table = pa.table({"step": train_loss_steps, "train_loss": train_losses})
                pq.write_table(train_loss_table, os.path.join(paths_config["trained_models_folder"], f"{preset}_s{seed}_train_loss.parquet"), compression='brotli')
            if test_losses is not None:
                test_loss_table = pa.table({"step": test_loss_steps, "test_loss": test_losses})
                pq.write_table(test_loss_table, os.path.join(paths_config["trained_models_folder"], f"{preset}_s{seed}_test_loss.parquet"), compression='brotli')
