import signal
import optax
import sys
import os
import time
import pickle
import jax.random as jr
import equinox as eqx

from rvsr import RVSR
from train_utils import SuperresolutionTrainer, eval_inference_time
from data_utils import load_data_array
from typing import List
from paths_config import paths_config

def merge_dicts(original, override):
    """Recursively merges two dictionaries, giving priority to the override dictionary.
    Nested dictionaries can be replaced using the key "<replace>" with a dictionary of dictionaries (other types are tolerated) to be replaced.
    Args:
        original: The original dictionary.
        override: The override dictionary.
    Returns:
        The merged dictionary.
    """
    merged = original.copy()
    if "<replace>" in override:
        merged = {
            **merged,
            **override["<replace>"]
        }
    for key, value in override.items():
        if key != "<replace>":
            if isinstance(value, dict) and key in merged:
                merged[key] = merge_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged

# Default hyperparameters
hpars = {
    "folder": os.path.join(paths_config["train_runs_folder"], "test"),
    "protect_existing": True,
    "sr_rate": 4,
    "model_type": RVSR,
    "model_hpars": {
        "hidden_channels": 16,
        "conv_padding_method": "lp",
        "upscale_padding_method": "lp",
        "padding_method_kwargs": {
            "length": 4,
            "width": 5,
            "apodization": "tukey",  
            "correlate_method": "fft",
            "cholesky_stab": 1e-7
        },
        "output_crop": 0
    },
    "image_shape": (3, 192, 192),
    "batch_size": 64,
    "shuffle_train_data": True,
    "optimizer": optax.adam,
    "optimizer_kwargs": {
        "eps": 1e-3,
        "b1": 0.9,
        "b2": 0.999,
        "eps_root": 0.0,
        "nesterov": False
    },
    "lr_schedule": optax.piecewise_interpolate_schedule,
    "lr_schedule_kwargs": {
        "interpolate_type": "linear",
        "init_value": 5e-6,
        "boundaries_and_scales": {
            100: 2800, # Increase to 0.014
            1_000_000: 1, # Keep at 0.014
            1_500_000: 0 # Ramp down to 0
        }
    },
    "train_image_save_interval": 0,
    "test_image_save_interval": 1000,
    "checkpoint_steps": [*range(0, 1_000, 100), *range(1_000, 10_000, 1_000), *range(10_000, 100_000, 10_000), *range(100_000, 1_500_000, 100_000), 1_500_000],
    "seed": 0,
    "train_steps": 1_500_000,
    "test_data_path": os.path.join(paths_config["dataset_folder"], paths_config["test_data_filename"]),
    "train_data_path": os.path.join(paths_config["dataset_folder"], paths_config["train_data_filename"]),
    "save": True,
    "timing_filename": None,
}

presets = {
    # Linear prediction padding, 1x1 neighborhood
    "lp1x1cs": {  # No apodization, stabilized covariance method
        "model_hpars": {
            "padding_method_kwargs": {"length": 1, "width": 1, "correlate_method": "cov_stab", "apodization": None}
        }
    },
    "lp1x1cs_oc1": {  # No apodization, stabilized covariance method
        "compat_preset": "lp1x1cs_el1",
        "model_hpars": {
            "padding_method_kwargs": {"length": 1, "width": 1, "correlate_method": "cov_stab", "apodization": None},
            "output_crop": 1
        }
    },

    # Linear prediction padding, 2x1 neighborhood
    "lp2x1": {  # Tukey apodization, direct autocorrelation method
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 1, "correlate_method": "direct"}
        }
    },
    "lp2x1cs": {  # No apodization, stabilized covariance method
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 1, "correlate_method": "cov_stab", "apodization": None}
        }
    },
    "lp2x1cs_oc1": {  # No apodization, stabilized covariance method
        "compat_preset": "lp2x1cs_el1",
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 1, "correlate_method": "cov_stab", "apodization": None},
            "output_crop": 1
        }
    },

    # Linear prediction padding, 2x3 neighborhood
    "lp2x3": {  # Tukey apodization, direct autocorrelation method
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 3, "correlate_method": "direct"}
        }
    }, 
    "lp2x3_oc1": {  # Tukey apodization, direct autocorrelation method, output crop 1
        "compat_preset": "lp2x3_el1",
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 3, "correlate_method": "direct"},
            "output_crop": 1
        }
    },
    "lp2x3_oc5": {  # Tukey apodization, direct autocorrelation method, output crop 5
        "compat_preset": "lp2x3_el50",
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 3, "correlate_method": "direct"},
            "output_crop": 5
        }
    },

    # Linear prediction padding, 2x5 neighborhood
    "lp2x5": {  # Tukey apodization, FFT autocorrelation method
        "model_hpars": {
            "padding_method_kwargs": {"length": 2, "width": 5}
        }
    },

    # Linear prediction padding, 3x3 neighborhood
    "lp3x3": {  # Tukey apodization, FFT autocorrelation method
        "model_hpars": {
            "padding_method_kwargs": {"length": 3, "width": 3}
        }
    },

    # Linear prediction padding, 4x5 neighborhood
    "lp4x5": { # Tukey apodization, FFT autocorrelation method
        "model_hpars": {
            "padding_method_kwargs": {"length": 4, "width": 5}
        }
    },

    # Linear prediction padding, 6x7 neighborhood
    "lp6x7": { # Tukey apodization, FFT autocorrelation method
        "model_hpars": {
            "padding_method_kwargs": {"length": 6, "width": 7}}
    },

 
    # Zero padding
    "zero-repl": {
        "compat_preset": "zero",
        "model_hpars": {
            "conv_padding_method": "zero",
            "upscale_padding_method": None,
        }
    },
    "zero_oc1": {  # output crop 1
        "compat_preset": "zero_el1",
        "model_hpars": {
            "conv_padding_method": "zero",
            "upscale_padding_method": None,
            "output_crop": 1
        }
    },
    "zero_oc5": {  # output crop 5
        "compat_preset": "zero_el50",
        "model_hpars": {
            "conv_padding_method": "zero",
            "upscale_padding_method": None,
            "output_crop": 5
        }
    },

    "zero-zero": { 
        "compat_preset": "zerozero",
        "model_hpars": {
            "conv_padding_method": "zero",
            "upscale_padding_method": "zero",
        }
    },

    # replicate pad
    "repl": { 
        "model_hpars": {
            "conv_padding_method": "repl",
            "upscale_padding_method": None,
        }
    },
    "repl_oc1": { # output crop 1
        "compat_preset": "repl_el1",
        "model_hpars": {
            "conv_padding_method": "repl",
            "upscale_padding_method": None,
            "output_crop": 1
        }
    }, 
    "repl_oc5": { # output crop 5
        "compat_preset": "repl_el50",
        "model_hpars": {
            "conv_padding_method": "repl",
            "upscale_padding_method": None,
            "output_crop": 5
        }
    }, 

    #
    # polynomial pad
    #
    "extr1": { 
        "model_hpars": {
            "conv_padding_method": "extr",
            "upscale_padding_method": "extr",
            "<replace>": {
                "padding_method_kwargs": {"num_predictors": 1}
            }
        }
    },
    "extr2": { 
        "model_hpars": {
            "conv_padding_method": "extr",
            "upscale_padding_method": "extr",
            "<replace>": {
                "padding_method_kwargs": {"num_predictors": 2}
            }
        }
    },
    "extr3": { 
        "model_hpars": {
            "conv_padding_method": "extr",
            "upscale_padding_method": "extr",
            "<replace>": {
                "padding_method_kwargs": {"num_predictors": 3}
            }
        }
    }
}

# Get the hyperparameter dict for a preset
def get_preset_hpars(preset, seed = None):
    if preset not in presets.keys():
        raise ValueError(f"Preset {preset} has not been defined")
    preset_hpars = merge_dicts(hpars, presets[preset])
    preset_hpars["preset"] = preset
    if seed is not None:
        preset_hpars["seed"] = seed
    if "seed" in preset_hpars:
        preset_hpars["folder"] = os.path.join(paths_config["train_runs_folder"], f"{preset}_s{preset_hpars['seed']}")
    return preset_hpars

if __name__ == "__main__":
    # mapping between accepted command line argument names and the desired datatype
    # eg. you can call 'python3 job.py seed=42'
    # special non-hpars arguments: train_steps and preset
    arg_dtypes = {
        "seed": int,
        "resume_checkpoint": int,
        "folder": str,
        "train_data_path": str,
        "test_data_path": str,
        "train_steps": int,
        "batch_size": int,
        "protect_existing": bool,
        "save": bool,
        "checkpoint_steps": List[int],
        "timing_filename": str,
        "inference_steps": int,
        "inference_data_path": str,
        "cooldown": int
    }

    args = {}
    for arg in sys.argv[1:]:
        name, val = arg.split("=")
        args[name] = val
    if "preset" in args:
        preset = args["preset"]
        print(f"preset={preset}")
        hpars = get_preset_hpars(preset, args["seed"] if "seed" in args else None)
        del args["preset"]
    # Preset hpars can be overridden using args
    for name, val in args.items():
        print(f"{name}={val}")
        if name not in arg_dtypes.keys():
            raise KeyError(f"No datatype specified in arg_dtypes for argument {name}")
        val_dtype = arg_dtypes[name]
        # add the argument to the args dict using correct dtype
        if val_dtype == bool:
            hpars[name] = val == "True"
        elif val_dtype == List[int]:
            hpars[name] = [int(x) for x in val[1:-1].split(",")]
        else:
            hpars[name] = val_dtype(val)

    if "inference_steps" not in hpars:
        trainer = SuperresolutionTrainer(**hpars)
        if "train_data" not in locals():
            if not os.path.exists(hpars["train_data_path"]):
                raise Exception(f"Can't find train data file {hpars["test_data_path"]} configured with train_data_path")
            train_data = load_data_array(hpars["train_data_path"], "train")
        if "test_data" not in locals():
            if hpars["test_data_path"] == hpars["train_data_path"]:
                test_data = train_data
            else:
                if not os.path.exists(hpars["test_data_path"]):
                    raise Exception(f"Can't find train data file {hpars["test_data_path"]} configured with test_data_path")
                test_data = load_data_array(hpars["test_data_path"], "test")

    if "cooldown" in hpars:
        time.sleep(hpars["cooldown"])

    try:
        if "inference_steps" in hpars:
            model, model_state = eqx.nn.make_with_state(hpars["model_type"])(sr_rate=hpars["sr_rate"], inference=True, **hpars["model_hpars"], key = jr.PRNGKey(42))
            timing = eval_inference_time(model, model_state, hpars, batch_size=hpars["batch_size"], num_batches=hpars["inference_steps"])
            if "timing_filename" in hpars and hpars["timing_filename"] is not None:
                with open(hpars["timing_filename"], "wb") as f:
                    pickle.dump(timing, f)
        else:
            trainer.train(train_data, test_data, hpars["train_steps"])
    except KeyboardInterrupt:
        signal.raise_signal(signal.SIGINT)
    except Exception as e:
        # Save exception text to file
        if hpars["save"] and "inference_steps" not in hpars:
            with open(os.path.join(hpars['folder'], "exception.txt"), "w") as f:
                f.write(str(e))
        raise e
    print()
