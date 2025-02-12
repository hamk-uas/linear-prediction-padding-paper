import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import os
import json
import datetime
import PIL.Image
import types
import importlib
import pickle
import time
import gc

from data_utils import np_linear_to_srgb, preprocess_batch_for_superresolution_task, ImageBatcher

# Get the train steps on which train loss will be calculated
def get_train_loss_steps(hpars):
    return np.arange(1, hpars["train_steps"] + 1)

# Get the train steps on which test loss will be calculated
def get_test_loss_steps(hpars):
    return sorted(list(set(hpars["checkpoint_steps"] + list(range(0, hpars["train_steps"], hpars["test_image_save_interval"])))))

# Calculate MSE
def ms_error(predictions, targets):
    return jnp.mean(optax.squared_error(predictions, targets))

# Get model predictions
@eqx.filter_jit
def infer(model, state, inputs, key=None):
    predictions, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(inputs, state, key)
    return predictions

# Evaluate GPU time used by inference
def eval_inference_time(model, model_state, hpars, batch_size, num_batches):
    elapsed_times = []
    key = jr.key(0)
    for batch_num in range(num_batches):
        key, use_key= jr.split(key)
        batch = jax.block_until_ready(jax.random.uniform(use_key, minval=-1.0, maxval=1.0, shape=(batch_size, hpars["image_shape"][0], hpars["image_shape"][1]//hpars["sr_rate"], hpars["image_shape"][2]//hpars["sr_rate"])))
        gc_old = gc.isenabled()
        gc.disable()
        start_time = time.perf_counter()
        predictions = jax.block_until_ready(infer(model, model_state, batch))
        end_time = time.perf_counter()
        if gc_old:
            gc.enable()
        elapsed_times.append(end_time - start_time)
    return elapsed_times

# Create and save sample images 
def make_images(inputs, targets, predictions, hpars, num=None, save_path=None):
    if num is None:
        num = inputs.shape[0]
    upscaled_inputs = jax.vmap(lambda x: jax.image.resize(x, (x.shape[0], x.shape[1]*4, x.shape[2]*4), "nearest"))(inputs[:num])
    pad = hpars["model_hpars"]["output_crop"]*hpars["sr_rate"]
    image = PIL.Image.fromarray(np.hstack(np.moveaxis(np_linear_to_srgb(np.concatenate((upscaled_inputs, targets[:num], np.pad(predictions[:num], ((0, 0), (0, 0), (pad, pad), (pad, pad)))), axis=2)*0.5+0.5), source=1, destination=3)))
    if save_path is not None:
        image.save(save_path)
    return image

# Custom encoder for JSON serialization
def custom_encoder(obj):
    # objects that are incompatible with pickle must be serialized separately
    if isinstance(obj, type) or isinstance(obj, types.FunctionType):
        return f"<attr>{obj.__name__}<from>{obj.__module__}"
    elif isinstance(obj, dict):
        copy = {}
        for key, value in obj.items():
            if isinstance(key, int):
                copy[f"<int>{key}"] = custom_encoder(value)
            elif isinstance(key, float):
                copy[f"<float>{key}"] = custom_encoder(value)
            else:
                copy[key] = custom_encoder(value)
        return copy
    else:
        return obj

# Custom decoder for JSON serialization
def custom_decoder(obj):
    if isinstance(obj, str):
        if obj.startswith("<attr>"):
            module = importlib.import_module(obj[obj.index("<from>")+6:])
            return getattr(module, obj[6:obj.index("<from>")])
        else:
            return obj
    elif isinstance(obj, dict):
        copy = {}
        for key, value in obj.items():
            if key.startswith("<int>"):
                copy[int(key[5:])] = custom_decoder(value)
            elif key.startswith("<float>"):
                copy[float(key[7:])] = custom_decoder(value)
            else:
                copy[key] = custom_decoder(value)
        return copy
    else:
        return obj

# JSON encode dict using custom encoder
def json_encode_dict(data):
    return json.dumps(custom_encoder(data), indent=4)

# JSON decode dict using custom decoder
def json_decode_dict(json_string):
    return custom_decoder(json.loads(json_string))

# Compute gradients
@eqx.filter_value_and_grad(has_aux=True)
def compute_grads(model, state, targets, inputs, hpars, key):
    predictions, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(inputs, state, key)
    # Compat mode:
    # crop = hpars["sr_rate"]*hpars["model_hpars"]["output_crop"]
    # loss = ms_error(predictions, targets[..., crop:targets.shape[-2]-crop, crop:targets.shape[-1]-crop])    
    loss = ms_error(predictions, targets)
    return loss, (state, predictions)

# Compute predictions for a center-cropped batch and calculate loss
@eqx.filter_jit
def jitted_test(model, state, batch, hpars):
    key = jr.PRNGKey(0)            
    inputs, targets = preprocess_batch_for_superresolution_task( # no random crop for tests
        batch, hpars["image_shape"][1], hpars["image_shape"][2], hpars["sr_rate"], False, key)            
    predictions, _ = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(inputs, state, key)
    # Compat mode:
    #crop = hpars["sr_rate"]*hpars["model_hpars"]["output_crop"]
    #loss = ms_error(predictions, targets[..., crop:targets.shape[-2]-crop, crop:targets.shape[-1]-crop])
    loss = ms_error(predictions, targets)
    return predictions, loss, inputs, targets

# Utility class for super-resolution training
class SuperresolutionTrainer():
    history_attrs = ["loss_steps", "losses", "learning_rates"]
    test_history_attrs = ["test_loss_steps", "test_losses"]
    # list of attributes that take up a lot of space and need to be handled separately

    def save_history(self):
        loss_path = os.path.join(self.hpars['folder'], "train_history.csv")
        np.savetxt(loss_path, np.vstack(
            [getattr(self, key) for key in SuperresolutionTrainer.history_attrs]).T,
            delimiter=',', header=",".join(SuperresolutionTrainer.history_attrs))
        
        loss_path = os.path.join(self.hpars['folder'], "test_history.csv")
        np.savetxt(loss_path, np.vstack(
            [getattr(self, key) for key in SuperresolutionTrainer.test_history_attrs]).T,
            delimiter=',', header=",".join(SuperresolutionTrainer.test_history_attrs))

    def save_hpars(self):
        # Dump hpars as json
        hpars_path = os.path.join(self.hpars['folder'], "hpars.json")
        with open(hpars_path, 'wt') as f:
            json.dump(custom_encoder(self.hpars), f, indent=4)

    def save(self):
        self.hpars["resume_checkpoint"] = self.step
        self.save_hpars()
        self.save_history()

        # Create checkpoint folder
        checkpoint_folder = os.path.join(self.hpars['folder'], str(self.step).zfill(8))
        os.makedirs(checkpoint_folder, exist_ok=True)

        #Backup and temporarily del attrs that cannot be pickled
        no_pickle_attrs = (
            ["lr_schedule", "optimizer", "hpars"] +
            SuperresolutionTrainer.history_attrs +
            SuperresolutionTrainer.test_history_attrs
        )
        attr_backup = {}
        for key in no_pickle_attrs:
            attr_backup[key] = getattr(self, key)
            delattr(self, key)

        # Pickle self
        trainer_path = os.path.join(checkpoint_folder, "trainer.pickle")
        with open(trainer_path, 'wb') as f:
            pickle.dump(self, f)

        # Restore attrs from backup
        for key in no_pickle_attrs:
            setattr(self, key, attr_backup[key])
            del attr_backup[key]

    def init_no_pickle_attrs(self):
        self.lr_schedule = self.hpars["lr_schedule"](**self.hpars["lr_schedule_kwargs"])
        self.optimizer = optax.inject_hyperparams(self.hpars["optimizer"])(learning_rate=self.lr_schedule, **self.hpars["optimizer_kwargs"])

    def test(self, test_data, save_images=False):
        test_batcher = ImageBatcher(self.hpars["batch_size"], False)
        num_steps = len(test_data)//self.hpars["batch_size"]
        example_indexes = [2,7,24,36,38,52,53]
        inputs_list, targets_list, predictions_list = [], [], []
        losses = []
        for i in range(num_steps):
            batch = jnp.array(test_batcher.get_batch(test_data))
            predictions, loss, inputs, targets = jitted_test(self.model, self.model_state, batch, self.hpars)
            if save_images:
                for idx in example_indexes:
                    if idx < self.hpars["batch_size"]*(i+1) and idx >= self.hpars["batch_size"]*i:
                        inputs_list.append(inputs[idx - (self.hpars["batch_size"]*i)])
                        targets_list.append(targets[idx - (self.hpars["batch_size"]*i)])
                        predictions_list.append(predictions[idx - (self.hpars["batch_size"]*i)])
            losses.append(loss)
        self.test_losses.append(np.mean(losses))
        self.test_loss_steps.append(self.step)
        if save_images:
            image_path = os.path.join(self.hpars['folder'], "test_images", f"{str(self.step).zfill(8)}.png")
            make_images(np.stack(inputs_list), np.stack(targets_list), np.stack(predictions_list), self.hpars, save_path=image_path)
    
    def __init__(self, **hpars):
        # Store hpars to self
        self.hpars = hpars

        # Ensure default folder is the present working directory
        if "folder" not in self.hpars:
            self.hpars["folder"] = "."

        if "resume_checkpoint" in self.hpars and self.hpars["resume_checkpoint"] == -1:
            # Get a list of subfolder names
            subfolders = [int(x) for x in os.listdir(self.hpars["folder"]) if x.isdigit()]
            # Numerically sort subfolders
            subfolders.sort()
            if len(subfolders) > 0:
                # Check if the largest number subfolder contains a pickle
                if os.path.exists(os.path.join(self.hpars['folder'], str(subfolders[-1]).zfill(8), "trainer.pickle")):
                    self.hpars["resume_checkpoint"] = subfolders[-1]
                elif len(subfolders) > 1 and os.path.exists(os.path.join(self.hpars['folder'], str(subfolders[-2]).zfill(8), "trainer.pickle")):
                    self.hpars["resume_checkpoint"] = subfolders[-2]
                else:
                    print("Bad checkpoints. Starting from beginning.")
                    del self.hpars["resume_checkpoint"]
            else:
                print("No checkpoints. Starting from beginning.")
                del self.hpars["resume_checkpoint"]
            
        if "resume_checkpoint" in self.hpars:
            # Resuming checkpoint  
            print(f"Resuming checkpoint {str(self.hpars['resume_checkpoint']).zfill(8)}")
            hpars_path = os.path.join(self.hpars['folder'], "hpars.json")
            checkpoint_folder = os.path.join(self.hpars['folder'], str(self.hpars['resume_checkpoint']).zfill(8))
            trainer_path = os.path.join(checkpoint_folder, "trainer.pickle")
            hpars = self.hpars.copy()  # Ensure we don't load hpars from pickled file
            with open(trainer_path, 'rb') as f:
                copy = pickle.load(f)
            self.__dict__.update(copy.__dict__)
            with open(hpars_path, 'rt') as f:
                self.hpars = {
                    **custom_decoder(json.load(f)),
                    **hpars  # Merge hpars
                }
            # Initialize no-pickle attrs
            self.init_no_pickle_attrs()

            # Load CSVs
            history_arr = np.loadtxt(os.path.join(self.hpars['folder'], "train_history.csv"), delimiter=",", ndmin=2).T
            test_history_arr = np.loadtxt(os.path.join(self.hpars['folder'], "test_history.csv"), delimiter=",", ndmin=2).T

            # Forget history rows recorded after the checkpoint step
            history_arr = history_arr[:, history_arr[0] <= self.hpars['resume_checkpoint']]
            test_history_arr = test_history_arr[:, test_history_arr[0] <= self.hpars['resume_checkpoint']]

            # Fix overlapping data by letting later duplicates replace earlier ones
            max_step = -1
            row = 0
            while row < history_arr.shape[1]:
                row_step = history_arr[0, row]
                if row_step > max_step:
                    max_step = history_arr[0, row]
                else:
                    history_arr = np.hstack((history_arr[:, :row][:, history_arr[0, :row] < row_step], history_arr[:, row:]))
                row += 1
            max_step = -1
            row = 0
            while row < test_history_arr.shape[1]:
                row_step = test_history_arr[0, row]
                if row_step > max_step:
                    max_step = test_history_arr[0, row]
                else:
                    test_history_arr = np.hstack((test_history_arr[:, :row][:, test_history_arr[0, :row] < row_step], test_history_arr[:, row:]))
                row += 1
            
            # Support adding more history_attrs and test_history_attrs (only at the end), fill history with NaN
            history_arr = np.pad(history_arr, ((0, len(SuperresolutionTrainer.history_attrs) - history_arr.shape[0]), (0, 0)), mode="constant", constant_values=np.nan)
            test_history_arr = np.pad(test_history_arr, ((0, len(SuperresolutionTrainer.test_history_attrs) - test_history_arr.shape[0]), (0, 0)), mode="constant", constant_values=np.nan)

            for attr_name, vals in zip(SuperresolutionTrainer.history_attrs, history_arr):
                self.__setattr__(attr_name, vals.tolist())
            for attr_name, vals in zip(SuperresolutionTrainer.test_history_attrs, test_history_arr):
                self.__setattr__(attr_name, vals.tolist())

        else:
            # Init from scratch using hpars
            # Add creation datetime to hpars
            now = datetime.datetime.now()
            self.hpars["creation_datetime"] = f"{now.date()}-{now.time().hour:02d}-{now.time().minute:02d}"
            # Protect existing training run folders (configurable)
            if self.hpars["protect_existing"]:
                if os.path.exists(self.hpars["folder"]):
                    raise Exception(f"The folder '{self.hpars['folder']}' already exists. Aborting.")
            # Create folders
            if self.hpars["save"]: 
                os.makedirs(self.hpars['folder'], exist_ok=True)
                os.makedirs(os.path.join(self.hpars['folder'], "test_images"), exist_ok=True)
                os.makedirs(os.path.join(self.hpars['folder'], "train_images"), exist_ok=True)
            # Initialize no-pickle attrs
            self.init_no_pickle_attrs()
            # Initialize JAX random number generator
            self.key = jr.PRNGKey(self.hpars["seed"])
            self.key, subkey = jr.split(self.key)
            # Initialize Numpy random number generator
            self.rng = np.random.default_rng(seed=np.array(subkey))
            # Initialize training data batcher
            self.image_batcher = ImageBatcher(self.hpars["batch_size"], self.hpars["shuffle_train_data"], self.rng)        
            # Initialize model
            self.key, subkey = jr.split(self.key)
            self.model, self.model_state = eqx.nn.make_with_state(self.hpars["model_type"])(sr_rate = self.hpars["sr_rate"], **self.hpars["model_hpars"], key = subkey)
            # Initialize optimizer
            self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
            # Initialize histories
            self.loss_steps = []
            self.losses = []
            self.learning_rates = []  # Stored at density of training loss
            self.test_loss_steps = []
            self.test_losses = []
            # Step count
            self.step = 0
            self.checkpoint_steps_index = 0
            if self.hpars["save"]:
                self.save_hpars()

    # Training step
    @eqx.filter_jit
    def train_step(self, model, state, batch, opt_state, key):
        # Preprocess batch
        key, subkey = jr.split(key)
        inputs, targets = preprocess_batch_for_superresolution_task(batch, self.hpars["image_shape"][1], self.hpars["image_shape"][2], self.hpars["sr_rate"], True, subkey)
        # Inference, grads, losses
        key, subkey = jr.split(key)
        (loss, (state, predictions)), grads = compute_grads(model, state, targets, inputs, self.hpars, subkey)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, targets, predictions, loss, inputs, key

    def train(
        self,
        train_data,
        test_data,
        end_step: int
    ):
        if self.hpars["timing_filename"] is not None:
            timing = []

        # Fix missing current test loss from buggy earlier runs
        if len(self.test_loss_steps) > 0 and self.test_loss_steps[-1] < self.step:
            self.test(test_data, self.hpars["save"])
            if self.hpars["save"]:
                self.save()

        if self.step == 0 and end_step > self.step:
            # Write test images and record test loss
            self.test(test_data, self.hpars["save"])
        while self.step < end_step:
            batch = jnp.array(self.image_batcher.get_batch(train_data))
            if self.hpars["timing_filename"] is None:
                self.model, self.model_state, self.opt_state, targets, predictions, loss, inputs, self.key = self.train_step(
                    self.model, self.model_state, batch, self.opt_state, self.key)
            else:
                start_time = time.perf_counter()
                self.model, self.model_state, self.opt_state, targets, predictions, loss, inputs, self.key = jax.block_until_ready(
                    self.train_step(self.model, self.model_state, batch, self.opt_state, self.key)
                )
                end_time = time.perf_counter()
                timing.append(end_time - start_time)
            self.step += 1
            if jnp.isnan(loss):
                raise ValueError("Training diverged, loss became nan")
            
            # Update histories
            self.loss_steps.append(self.step)
            self.losses.append(loss)
            self.learning_rates.append(self.opt_state.hyperparams['learning_rate'])

            # Checkpointing and drawing
            while self.hpars["checkpoint_steps"][self.checkpoint_steps_index] < self.step and self.checkpoint_steps_index < len(self.hpars["checkpoint_steps"]):
                # Update when is the next checkpoint
                self.checkpoint_steps_index += 1

            is_checkpoint_step = self.checkpoint_steps_index < len(self.hpars["checkpoint_steps"]) and self.hpars["checkpoint_steps"][self.checkpoint_steps_index] == self.step
            is_last_step = self.step == end_step
            if is_checkpoint_step or is_last_step:
                # We are at a checkpoint step or at the end of the training run
                # Save checkpoint
                self.test(test_data, self.hpars["save"])
                if self.hpars["save"]:
                    self.save()
                image_path = os.path.join(self.hpars['folder'], "train_images", f"{str(self.step).zfill(8)}.png")
                make_images(inputs, targets, predictions, self.hpars, 8, save_path=image_path if self.hpars["save"] else None)
            else: 
                if self.hpars["train_image_save_interval"] != 0 and self.step % self.hpars["train_image_save_interval"] == 0:
                    # Save train images
                    image_path = os.path.join(self.hpars['folder'], "train_images", f"{str(self.step).zfill(8)}.png")
                    make_images(inputs, targets, predictions, self.hpars, 8, save_path=image_path if self.hpars["save"] else None)
                if self.hpars["test_image_save_interval"] != 0 and self.step % self.hpars["test_image_save_interval"] == 0:
                    # Record test loss and save test images
                    self.test(test_data, save_images=self.hpars["save"])
            test_loss_str = "" if len(self.test_losses) == 0 else f", test loss={self.test_losses[-1]:.5f} (step {int(self.test_loss_steps[-1])})"
            if (loss > 1e6) and (self.step > 100):
                raise RuntimeError(f"Training aborted due to divergent loss at step {self.step}. Loss value: {loss}")
            if (self.step <= 100) or (self.step % 100 == 0):
                print((
                    f"step: {self.step}/{end_step}, " +
                    f"loss: {loss:.5f}, " +
                    f"lr={self.learning_rates[-1]:.5f}{test_loss_str}"
                    ).ljust(165),
                    end="\r"
                )
        if self.hpars["timing_filename"] is not None:
            with open(self.hpars["timing_filename"], "wb") as f:
                pickle.dump(timing, f)