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

num_seeds = 12  # Seeds run from 0 to num_seeds-1 inclusive
output_crop_step = 4  # 4 or hpars["sr_rate"] for input resolution, 1 for output resolution
num_output_crops = 11  # 11 if output_crop_step == 4, to calculate edge losses and center loss (theoretical receptive field not seeing edge)
results_filename = os.path.join(paths_config["results_folder"], paths_config["eval_results_filename"])
data_id = "test"  # "train" or "test" data
data_path = os.path.join(paths_config["dataset_folder"], paths_config["test_data_filename"])
batch_size = 500 # 64 to simulate training-time behavior, or 50 (or 500) to use all images in train or test set

def weighted_elementwise_ms_error(predictions, targets, weight):
    return jnp.sum(optax.squared_error(predictions, targets)*weight, axis=(1, 2, 3))

@eqx.filter_jit
def infer(model, state, inputs, key=None):
    predictions, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(inputs, state, key)
    return predictions

def get_masks(hpars, include_channel_dimension=True):
    if include_channel_dimension:
        masks = np.zeros((num_output_crops, 1, 1, hpars["image_shape"][1], hpars["image_shape"][2])) # (output_crop, batch, channel, y, x)
        for output_crop in range(num_output_crops):
            masks[output_crop, 0, 0, output_crop*output_crop_step:hpars["image_shape"][1]-output_crop*output_crop_step, output_crop*output_crop_step:hpars["image_shape"][2]-output_crop*output_crop_step] = 1.0
    else:
        masks = np.zeros((num_output_crops, 1, hpars["image_shape"][1], hpars["image_shape"][2])) # (output_crop, batch, channel, y, x)
        for output_crop in range(num_output_crops):
            masks[output_crop, 0, output_crop*output_crop_step:hpars["image_shape"][1]-output_crop*output_crop_step, output_crop*output_crop_step:hpars["image_shape"][2]-output_crop*output_crop_step] = 1.0
    return masks

def get_mask_sums(hpars):
    return np.sum(get_masks(hpars), axis=(-4, -3, -2, -1)) # (output_crop)

def evaluate_sumsqs(model, model_state, batch_size=batch_size):
    output_crop_sumsqs_total = [[] for x in range(num_output_crops)]
    masks = get_masks(hpars, False)
    for batch_num in range(len(data)//batch_size):
        batch = jnp.array(data[batch_num*batch_size:(batch_num + 1)*batch_size])
        #print(batch.shape) #(64, 3, 512, 512)
        window_size = (hpars["image_shape"][1] + 2*hpars["sr_rate"], hpars["image_shape"][2] + 2*hpars["sr_rate"])
        #print(window_size) #(200, 200)
        sses = jnp.zeros((batch_size, hpars["image_shape"][1], hpars["image_shape"][2]))
        num_shifts = 0
        # Crop a window at multiple offsets distributed on a grid
        N = 2 # Number of grid points in each spatial dimension
        for shift_y in [(batch.shape[2] - window_size[0])*y//(N-1) for y in range(N)]:
            for shift_x in [(batch.shape[3] - window_size[1])*x//(N-1) for x in range(N)]:
                inputs, targets = train_utils.preprocess_batch_for_superresolution_task(
                    batch[:, :, shift_y:shift_y+window_size[0], shift_x:shift_x+window_size[1]], hpars["image_shape"][1], hpars["image_shape"][2], hpars["sr_rate"], False, None
                )
                predictions = infer(model, model_state, inputs)
                sses = sses + jnp.mean((predictions - targets)**2, axis=1) # Mean over channels, accumulate
                num_shifts += 1
        sses = sses / num_shifts
        output_crop_sumsqs = jnp.sum(sses*masks, axis=(-2, -1))  # Sum over y and x --> (batch)
        #print(output_crop_sumsqs.shape) #(11, 25) (output_crop, batch)
        for output_crop in range(num_output_crops):
            #print(batch_sumsqs.shape) # (11, 25) (batch)
            output_crop_sumsqs_total[output_crop] += [float(x) for x in list(output_crop_sumsqs[output_crop])]
    return np.array(output_crop_sumsqs_total) #(output_crop, dataset)

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

    if "output_crop_sumsqs" not in results:
        results["output_crop_sumsqs"] = {}

    #results["output_crop_sumsqs"] = {}  # Reset output_crop_sumsqs in results (optional!)

    data = data_utils.load_data_array(data_path, id=data_id)

    for preset in presets:
        hpars = get_preset_hpars(preset)
        hpars["model_hpars"]["output_crop"] = 0  # Do not crop output so that we can measure MSE for all output crops simultaneously
        preset_mean_mses = []
        for seed in range(num_seeds):
            print(f"Preset: {preset}, seed: {seed}")
            model_path = os.path.join(paths_config["trained_models_folder"], f"{preset}_s{seed}.eqx")
            if os.path.exists(model_path):
                if preset not in results["output_crop_sumsqs"]:
                    results["output_crop_sumsqs"][preset] = {}
                if seed not in results["output_crop_sumsqs"][preset]:
                    results["output_crop_sumsqs"][preset][seed] = {}
                    model, model_state = eqx.nn.make_with_state(hpars["model_type"])(sr_rate=hpars["sr_rate"], **hpars["model_hpars"], key = jr.PRNGKey(42))
                    model_params, model_static = eqx.partition(model, eqx.is_array)
                    model_leaves, model_structure = jax.tree.flatten(model_params)
                    model_leaves = eqx.tree_deserialise_leaves(model_path, like=model_leaves)
                    params = jax.tree.unflatten(model_structure, model_leaves)
                    model = eqx.combine(params, model_static)
                    results["output_crop_sumsqs"][preset][seed] = evaluate_sumsqs(model, model_state)
                preset_mean_mses.append(np.mean(results["output_crop_sumsqs"][preset][seed], axis=1)/get_mask_sums(hpars))
                print(f"Mean MSE: ", *[f" {x:.7f}" for x in preset_mean_mses[-1].tolist()])
            else:
                print(f"No trained model found at {model_path}")
        
        with open(results_filename, "wb") as f:
            pickle.dump(results, f)

        print(f"Preset: {preset} summary")
        if len(preset_mean_mses) > 0:
            preset_mean_mses = np.mean(np.array(preset_mean_mses), axis=0)
            print(f"Mean MSE: ", *[f" {x:.7f}" for x in preset_mean_mses.tolist()])
