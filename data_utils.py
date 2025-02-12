import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Convert linear intensity data (0..1) to have sRGB transfer function (0..255)
def np_linear_to_srgb(value):
    result = np.zeros_like(value)
    # Convert linear RGB values to sRGB format.
    mask = value <= 0.0031308
    result[mask] = value[mask] * 12.92
    result[~mask] = 1.055 * (value[~mask] ** (1.0 / 2.4)) - 0.055
    result = np.clip(result, 0.0, 1.0) * 255.0
    return result.astype(np.uint8)

# Load a numpy array from the given path and print basic info. id is typically "train" or "test"
def load_data_array(path, id="train"):
    with open(path, 'rb') as f:
        data = np.load(f)
        print(f"Loaded {id} data numpy array with shape {data.shape} and dtype {data.dtype}")
        return data

# A utility class to get batches of images from a dataset (numpy array). Optionally, each batch can be sampled randomly without replacement from the dataset
class ImageBatcher:
    def __init__(self, batch_size: int, shuffle: bool, rng = None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.current_index = 0

    def get_batch(self, data) -> np.ndarray:
        if self.shuffle:
            batch = self.rng.choice(data, size=self.batch_size, replace=False, shuffle=False)
        else:
            end_index = self.current_index + self.batch_size
            if end_index > len(data):
                batch = np.concatenate((data[self.current_index:], data[:end_index - len(data)]))
                self.current_index = end_index % len(data)
            else:
                batch = data[self.current_index:end_index]
                self.current_index = end_index
        return batch
    
# Randomly crop a batch of images using JAX
def random_crop_jax(batch, height, width, key):
    y_max = batch.shape[2] - height
    x_max = batch.shape[3] - width

    key_y, key_x = jr.split(key)
    # Generate random crop coordinates
    y_start = jr.randint(key=key_y, shape=(len(batch),), minval=0, maxval=y_max + 1)
    x_start = jr.randint(key=key_x, shape=(len(batch),), minval=0, maxval=x_max + 1)

    # Create a list of start indices for each image in the batch
    start_indices = jnp.stack((y_start, x_start), axis=1)

    # Use vmap to apply dynamic_slice to each image in the batch
    cropped_images = jax.vmap(lambda image, start_index: jax.vmap(jax.lax.dynamic_slice, in_axes=(0, None, None))(image, start_index, (height, width)))(batch, start_indices)
    return cropped_images

# Center-crop a batch of images (JAX or numpy array)
def center_crop(image_batch, height, width):
    y_start = (image_batch.shape[2] - height)//2
    x_start = (image_batch.shape[3] - width)//2
    cropped_images = image_batch[:,:,y_start:y_start+height,x_start:x_start+width]
    return cropped_images

# Preprocess a batch of images for the superresolution task
# The input height and width are for the target images. Low-res output images will be smaller by a factor sr_rate
# The input images must be at least of size height + 2*sr_rate and width + 2*sr_rate.
# The input images will be center cropped or randomly cropped, depending on the random_crop argument
def preprocess_batch_for_superresolution_task(batch, height, width, sr_rate, random_crop: bool, key = None):
    # Crop (with some extra for bilinear resize)
    if random_crop:
        batch = random_crop_jax(batch, height + 2*sr_rate, width + 2*sr_rate, key)
    else:
        batch = center_crop(batch, height + 2*sr_rate, width + 2*sr_rate)
    # Translation equivariant resize (removes the extra)
    inputs = jax.vmap(jax.vmap(lambda x: jax.image.resize(x, (x.shape[0]//sr_rate, x.shape[1]//sr_rate), "bilinear", antialias=True)[1:-1,1:-1]))(batch)
    # (Remove the extra also from targets)
    targets = batch[:,:,sr_rate:-sr_rate,sr_rate:-sr_rate]
    return inputs, targets