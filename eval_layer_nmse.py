import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import pickle
import jax.random as jr

import train_utils
import data_utils
from job import presets, get_preset_hpars
from paths_config import paths_config
from rvsr import load_rvsr_weights
from padding import Padding2dLayer

padding_presets = {
    "extr3": {
        "padding_mode": "extr",
        "padding_method_kwargs": {
            "num_predictors": 3
        }
    },
    "lp2x3": {
        "padding_mode": "lp",
        "padding_method_kwargs": {
            "length": 2, 
            "width": 3, 
            "correlate_method": "direct"
        }
    },
    "zero": {
        "padding_mode": "zero",
        "padding_method_kwargs": {}
    },
    "repl": {
        "padding_mode": "repl",
        "padding_method_kwargs": {}
    }
}
train_presets=["zero_oc5", "lp2x3_oc5"]
num_seeds = 12
results_filename = os.path.join(paths_config["results_folder"], paths_config["eval_results_filename"])
data_id = "test"  # "train" or "test" data
data_path = os.path.join(paths_config["dataset_folder"], paths_config["test_data_filename"])
batch_size = 50

@eqx.filter_jit
def infer(model, state, inputs, key=None):
    predictions, state, intermediates = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None, None), out_axes=(0, None, 0)
    )(inputs, state, key, True)
    return predictions, intermediates

def evaluate_nmses(model, model_state, batch_size, padding_mode, padding_method_kwargs):
    padded_size = 30
    mask = np.ones((padded_size, padded_size))
    mask[1:-1, 1:-1] = 0
    padding2dLayer = Padding2dLayer(((1, 1), (1, 1)), padding_mode, padding_method_kwargs)
    pad = jax.jit(jax.vmap(padding2dLayer))
    sum_var = np.zeros((11,))
    sum_mses = np.zeros((11,))
    num_mses = 0
    for batch_num in range(len(data)//batch_size):
        batch = jnp.array(data[batch_num*batch_size:(batch_num + 1)*batch_size])
        #print(batch.shape) #(64, 3, 512, 512)
        window_size = (hpars["image_shape"][1] + 2*hpars["sr_rate"], hpars["image_shape"][2] + 2*hpars["sr_rate"])
        #print(window_size) #(200, 200)
        # Crop a window at multiple offsets distributed on a grid
        N = 2 # Number of grid points in each spatial dimension
        for shift_y in [(batch.shape[2] - window_size[0])*y//(N-1) for y in range(N)]:
            for shift_x in [(batch.shape[3] - window_size[1])*x//(N-1) for x in range(N)]:
                inputs, targets = train_utils.preprocess_batch_for_superresolution_task(
                    batch[:, :, shift_y:shift_y+window_size[0], shift_x:shift_x+window_size[1]], hpars["image_shape"][1], hpars["image_shape"][2], hpars["sr_rate"], False, None
                )
                predictions, intermediates = infer(model, model_state, inputs)
                intermediate_mses = []
                intermediate_var = []
                for intermediate in [inputs, *intermediates, predictions]:
                    cropped = intermediate[
                        ..., 
                        (intermediate.shape[-2]-(padded_size - 2))//2:intermediate.shape[-2]-(intermediate.shape[-2]-(padded_size - 2))//2, 
                        (intermediate.shape[-1]-(padded_size - 2))//2:intermediate.shape[-1]-(intermediate.shape[-1]-(padded_size - 2))//2, 
                    ]
                    padded = pad(cropped)
                    target = intermediate[
                        ..., 
                        (intermediate.shape[-2]-padded_size)//2:intermediate.shape[-2]-(intermediate.shape[-2]-padded_size)//2, 
                        (intermediate.shape[-1]-padded_size)//2:intermediate.shape[-1]-(intermediate.shape[-1]-padded_size)//2, 
                    ]
                    batch_image_channel_means = jnp.sum(target*mask, axis=(-2, -1), keepdims=True)/np.sum(mask)
                    intermediate_var.append(jnp.mean(jnp.sum(((target - batch_image_channel_means)*mask)**2, axis=(-2, -1))/np.sum(mask)))
                    diff = padded - target
                    mse = jnp.mean(jnp.sum((diff*mask)**2, axis=(-2, -1))/np.sum(mask))
                    intermediate_mses.append(mse)
                intermediate_var = np.array(intermediate_var)
                intermediate_mses = np.array(intermediate_mses)
                sum_var = sum_var + intermediate_var
                sum_mses = sum_mses + intermediate_mses
                num_mses += 1  # Unused due to final normalization, but we count it anyhow
    nmses = sum_mses / sum_var
    #print(f"Data variance: {sum_var}")
    return nmses

if __name__ == "__main__":
    # Are there previous results?
    if os.path.exists(results_filename):
        # Yes, load from pickle
        with open(results_filename, "rb") as f:
            results = pickle.load(f)
    else:
        # No, create folder (if it doesn't exist) and create empty dict
        os.makedirs(paths_config["results_folder"], exist_ok=True)
        results = {}

    results["padding_seed_layer_nmses"] = {}

    data = data_utils.load_data_array(data_path, id=data_id)

    for train_preset in train_presets:
        hpars = get_preset_hpars(train_preset)
        hpars["model_hpars"]["output_crop"] = 10  # Valid conv only
        for seed in range(num_seeds):
            print(f"Train preset: {train_preset}, seed: {seed}")
            model_path = os.path.join(paths_config["trained_models_folder"], f"{train_preset}_s{seed}.eqx")
            if os.path.exists(model_path):
                for padding_preset in padding_presets.keys():
                    padding_mode = padding_presets[padding_preset]["padding_mode"]
                    padding_method_kwargs = padding_presets[padding_preset]["padding_method_kwargs"]
                    if train_preset not in results["padding_seed_layer_nmses"]:
                        results["padding_seed_layer_nmses"][train_preset] = {}
                    if padding_preset not in results["padding_seed_layer_nmses"][train_preset]:
                        results["padding_seed_layer_nmses"][train_preset][padding_preset] = []
                    model, model_state = eqx.nn.make_with_state(hpars["model_type"])(sr_rate=hpars["sr_rate"], **hpars["model_hpars"], key = jr.PRNGKey(42))
                    oc0_model, model_state = eqx.nn.make_with_state(hpars["model_type"])(sr_rate=hpars["sr_rate"], **{**hpars["model_hpars"], "output_crop":0}, key = jr.PRNGKey(42))
                    model = load_rvsr_weights(model, model_path, oc0_model)
                    nmses = evaluate_nmses(model, model_state, batch_size, padding_mode, padding_method_kwargs)
                    results["padding_seed_layer_nmses"][train_preset][padding_preset].append(nmses)
                    print(padding_preset, nmses)
            else:
                print(f"No trained model found at {model_path}")

    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
