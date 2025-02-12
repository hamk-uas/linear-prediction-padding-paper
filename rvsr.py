import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import typing
import einops
import typing

from padding import Padding2dLayer

# Save RVSR model weights as if the RVSR model used output_crop=0.
# Compatible with load_rvsr_weights function.
def save_rvsr_weights(model, model_eqx_filename):
    model_params = eqx.partition(model, eqx.is_array)[0]
    model_leaves = jax.tree.flatten(model_params)[0]
    # Convert model weights to same format as with output_crop=0
    if len(model_leaves[20].shape) == 5 and model_leaves[20].shape[0] == 8:
        # Correct format, do nothing
        None
    else:
        # Stack individual RepViT weights to vmapped ones
        for from_index in range(40, len(model_leaves)):
            to_index = 20 + (from_index - 40)%20
            if len(model_leaves[to_index].shape) <= len(model_leaves[from_index].shape):
                model_leaves[to_index] = jnp.expand_dims(model_leaves[to_index], axis=0)
            model_leaves[to_index] = jnp.vstack((model_leaves[to_index], jnp.expand_dims(model_leaves[from_index], axis=0)))
    model_leaves = model_leaves[:40]
    eqx.tree_serialise_leaves(model_eqx_filename, model_leaves)

# Load RVSR model weights. Creates a model that is identical to target_model, except for the weights.
# In oc0_model, you must provide a version of the RVSR model that uses output_crop=0. 
def load_rvsr_weights(target_model, model_eqx_filename, oc0_model):
    target_model_params, target_model_static = eqx.partition(target_model, eqx.is_array)
    target_model_leaves, target_model_structure = jax.tree.flatten(target_model_params)
    #for i in range(len(target_model_leaves)):
    #    print(i, target_model_leaves[i].shape)
    if len(target_model_leaves[20].shape) == 5 and target_model_leaves[20].shape[0] == 8:
        # Can load without modifications
        new_model_leaves = eqx.tree_deserialise_leaves(model_eqx_filename, like=target_model_leaves)
    else:
        # Split stacks to shorter stacks and individual weight arrays
        source_model_params = eqx.partition(oc0_model, eqx.is_array)[0]
        source_model_leaves = jax.tree.flatten(source_model_params)[0]
        source_model_leaves = eqx.tree_deserialise_leaves(model_eqx_filename, like=source_model_leaves)
        new_model_leaves = source_model_leaves[:20]
        if len(target_model_leaves[20].shape) == 5:
            num_stacked = target_model_leaves[20].shape[0]
            # Get partial stacks
            for from_index in range(20, 40):
                new_model_leaves.append(source_model_leaves[from_index][:num_stacked])
        else:
            num_stacked = 0
        # Get individual arrays
        for from_stack_index in range(num_stacked, 8):
            for from_index in range(20, 40):
                new_model_leaves.append(source_model_leaves[from_index][from_stack_index])
    new_params = jax.tree.unflatten(target_model_structure, new_model_leaves)
    new_model = eqx.combine(new_params, target_model_static)   
    return new_model

# RVSR and its subblocks
# JAX recreation of https://github.com/huai-chang/RVSR. RVSR article: https://arxiv.org/pdf/2404.16484v1
# This implementation supports:
# * Configurable output center cropping implemented by using valid conv on a number of the last layers.
# * Making intermediate outputs available
# * Training and inference modes, however no conversion between the two has been implemented

class RepConv(eqx.Module):
    same_pad: bool
    padding: Padding2dLayer
    layers: tuple
    # For training, layers tuple is:
    # (branch_1: eqx.nn.Sequential, branch_2: eqx.nn.Sequential, res_conv1x1: eqx.nn.Sequential, res_conv3x3: eqx.nn.Conv2d) 
    # For inference, layers tuple is:
    # (conv3x3: eqx.nn.Conv2d)
    inference: bool

    def __init__(self, n_feats, ratio=2, *, inference: bool, same_pad: bool, conv_padding_method, padding_method_kwargs, key):  # *, signifies keyword-only arguments
        self.same_pad = same_pad
        if (conv_padding_method == "zero" and inference) or self.same_pad == False:
            self.padding = None
        else:  
            self.padding = Padding2dLayer(((1, 1), (1, 1)), conv_padding_method, padding_method_kwargs)  # kernel_size=3 -> padding width 1
        self.inference = inference
        # We have not implemented conversion between inference and training mode, because non-trained coefficients suffice for inference time measurement
        if self.inference:
            conv3x3 = eqx.nn.Conv2d(n_feats, n_feats, 3, 1, ((1, 1), (1, 1)) if (conv_padding_method == "zero") and same_pad else 0, key=key)
            self.layers = (conv3x3,)
        else:
            keys = jr.split(key, 8)
            branch_1 = eqx.nn.Sequential([            
                eqx.nn.Conv2d(n_feats, n_feats*ratio, 1, 1, 0, key=keys[0]),         # expand
                eqx.nn.Conv2d(n_feats*ratio, n_feats*ratio, 3, 1, 0, key=keys[1]),   # feature
                eqx.nn.Conv2d(n_feats*ratio, n_feats, 1, 1, 0, key=keys[2]),         # reduce
            ])
            branch_2 = eqx.nn.Sequential([
                eqx.nn.Conv2d(n_feats, n_feats*ratio, 1, 1, 0, key=keys[3]),
                eqx.nn.Conv2d(n_feats*ratio, n_feats*ratio, 3, 1, 0, key=keys[4]),
                eqx.nn.Conv2d(n_feats*ratio, n_feats, 1, 1, 0, key=keys[5]),
            ])
            res_conv3x3 = eqx.nn.Sequential([
                eqx.nn.Conv2d(n_feats, n_feats, 3, 1, 0, key=keys[6])
            ])
            res_conv1x1 = eqx.nn.Conv2d(n_feats, n_feats, 1, 1, 0, key=keys[7])
            self.layers = (branch_1, branch_2, res_conv1x1, res_conv3x3)

    def __call__(self, x, key=None):
        if self.inference:
            conv3x3, = self.layers
            if self.padding is None:
                out = conv3x3(x)
            else:
                padded = self.padding(x)
                out = conv3x3(padded)
        else:
            branch_1, branch_2, res_conv1x1, res_conv3x3 = self.layers
            if self.same_pad:
                padded = self.padding(x)
                res_3 = res_conv3x3(padded)
                res_1 = res_conv1x1(x)
                skip = x
                branch_1_out = branch_1(padded)
                branch_2_out = branch_2(padded)
                out = branch_1_out + branch_2_out + skip + res_1 + res_3
            else:
                skip = x[:, 1:-1, 1:-1]
                res_3 = res_conv3x3(x)
                res_1 = res_conv1x1(skip)
                branch_1_out = branch_1(x)
                branch_2_out = branch_2(x)
                out = branch_1_out + branch_2_out + skip + res_1 + res_3
        return out

class FFN(eqx.Module):
    layers: eqx.nn.Sequential
    def __init__(self, N, activation_function, key):
        subkey1, subkey2 = jr.split(key, 2)
        self.layers = eqx.nn.Sequential([
            eqx.nn.Conv2d(N, N*2, 1, 1, 0, key=subkey1),
            eqx.nn.Lambda(activation_function), # the only nonlinearity in the entire architecture is in this block
            eqx.nn.Conv2d(N*2, N, 1, 1, 0, key=subkey2)
        ])
    def __call__(self, x, key=None):
        return self.layers(x) + x
    
class RepViT(eqx.Module):
    layers: eqx.nn.Sequential
    def __init__(self, N, *, inference: bool, same_pad: bool, conv_padding_method, padding_method_kwargs, activation_function, key):
        subkey1, subkey2 = jr.split(key, 2)
        self.layers = eqx.nn.Sequential([
            RepConv(N, inference=inference, same_pad=same_pad, conv_padding_method=conv_padding_method, padding_method_kwargs=padding_method_kwargs, key=subkey1),
            # <-- SqeezeExcite goes optionally here
            FFN(N, activation_function, key=subkey2)
        ])
    def __call__(self, x, key=None):
        return self.layers(x)

class BilinearUpscaleLayer(eqx.Module):
    factor: int
    output_crop: int
    upscale_padding_method: None | str
    padding_method_kwargs: dict

    def __init__(self, factor: int, output_crop: int, upscale_padding_method=None, padding_method_kwargs={}):
        assert output_crop >= 0 and output_crop <= 10, f"output_crop between 0 to 10 inclusive supported, was {output_crop}"
        self.factor = factor
        self.output_crop = output_crop
        self.upscale_padding_method = upscale_padding_method
        self.padding_method_kwargs = padding_method_kwargs
        
    def __call__(self, x, key=None):
        if self.output_crop > 0:
            pre_crop = self.output_crop - 1
            x = x[:, pre_crop:x.shape[1]-pre_crop, pre_crop:x.shape[2]-pre_crop]
            x = jax.image.resize(x, (x.shape[0], x.shape[1]*self.factor, x.shape[2]*self.factor), "bilinear")[:, self.factor:-self.factor, self.factor:-self.factor]
        else:
            if self.upscale_padding_method is None:
                x = jax.image.resize(x, (x.shape[0], x.shape[1]*self.factor, x.shape[2]*self.factor), "bilinear")           
            else:
                x = Padding2dLayer(((1, 1), (1, 1)), self.upscale_padding_method, self.padding_method_kwargs)(x)
                x = jax.image.resize(x, (x.shape[0], x.shape[1]*self.factor, x.shape[2]*self.factor), "bilinear")[:, self.factor:-self.factor, self.factor:-self.factor]
        return x


class PixelShuffle(eqx.Module):
    factor: int
    # c_out = c_in // factor**2
    # h_out = h_in * factor
    # w_out = w_in * factor
    def __call__(self, x, key=None):
        return einops.rearrange(
            x, '(c b1 b2) h w -> c (h b1) (w b2)', b1=self.factor, b2=self.factor
        )


class RVSR(eqx.Module):
    head: eqx.nn.Sequential
    tail: eqx.nn.Sequential
    upscale: BilinearUpscaleLayer
    body_repvits_same_pad: RepViT
    body_repvits_no_pad: list
    body_spatial_reduction: int

    def __init__(
        self,
        sr_rate: int,
        hidden_channels: int,
        inference: bool = False,
        output_crop: int = 0,
        *,
        conv_padding_method: str,
        upscale_padding_method: None | str,
        padding_method_kwargs: dict,
        activation_function: typing.Callable = jax.nn.gelu,
        key: jax.Array = None
    ):
        assert output_crop >= 0 and output_crop <= 10, f"output_crop between 0 to 10 inclusive supported, was {output_crop}"
        # hidden_channels: number of channels in the RepViT layers
        keys = jr.split(key, 11)
        if output_crop == 10:
            self.head = eqx.nn.Conv2d(3, hidden_channels, 3, 1, 0, key=keys[0])
        else:
            self.head = eqx.nn.Sequential([
                Padding2dLayer(((1, 1), (1, 1)), conv_padding_method, padding_method_kwargs),
                eqx.nn.Conv2d(3, hidden_channels, 3, 1, 0, key=keys[0])
            ])

        # Could do this but instead we use vmap to create the layers and implement scan in body() to run them
        #self.body = eqx.nn.Sequential([
        #    RepViT(hidden_channels, conv_padding_method=conv_padding_method, padding_method_kwargs=padding_method_kwargs, key=keys[i+1]) for i in range(8)
        #])
        make_repvit_same_pad = lambda key: RepViT(hidden_channels, inference=inference, same_pad=True, conv_padding_method=conv_padding_method, padding_method_kwargs=padding_method_kwargs, activation_function=activation_function, key=key)
        make_repvit_no_pad = lambda key: RepViT(hidden_channels, inference=inference, same_pad=False, conv_padding_method=conv_padding_method, padding_method_kwargs=padding_method_kwargs, activation_function=activation_function, key=key)
        if output_crop <= 1:
            self.body_repvits_same_pad = eqx.filter_vmap(make_repvit_same_pad)(keys[1:9])
            self.body_repvits_no_pad = None
            self.body_spatial_reduction = 0
        elif output_crop >= 9:
            self.body_repvits_same_pad = None
            self.body_repvits_no_pad = [make_repvit_no_pad(key) for key in keys[1:9]]
            self.body_spatial_reduction = 8
        else:
            num_same_pad = 8 - (output_crop - 1)
            self.body_repvits_same_pad = eqx.filter_vmap(make_repvit_same_pad)(keys[1:1 + num_same_pad])
            self.body_repvits_no_pad = [make_repvit_no_pad(key) for key in keys[1 + num_same_pad:9]]
            self.body_spatial_reduction = 8 - num_same_pad

        self.tail = eqx.nn.Sequential([
            RepConv(hidden_channels, inference=inference, same_pad=(output_crop == 0), conv_padding_method=conv_padding_method, padding_method_kwargs=padding_method_kwargs, key=keys[9]),
            eqx.nn.Conv2d(hidden_channels, 3*(sr_rate**2), 1, 1, 0, key=keys[10]),
            PixelShuffle(sr_rate)])
        self.upscale = BilinearUpscaleLayer(factor=sr_rate, output_crop=output_crop, upscale_padding_method=upscale_padding_method, padding_method_kwargs=padding_method_kwargs)

    def body(self, x, get_intermediates=False):
        def f(_x, _dynamic_body_repvit):
            body_repvit = eqx.combine(_dynamic_body_repvit, static_body_repvits)
            return body_repvit(_x), None
        if self.body_repvits_same_pad is not None:
            dynamic_body_repvits, static_body_repvits = eqx.partition(self.body_repvits_same_pad, eqx.is_array)
            x = jax.lax.scan(f, x, dynamic_body_repvits)[0]
        if self.body_repvits_no_pad is not None:
            if get_intermediates:
                intermediates = []
                for body_repvit in self.body_repvits_no_pad[:-1]:
                    x = body_repvit(x)
                    intermediates.append(x)
                x = self.body_repvits_no_pad[-1](x)
                return x, intermediates
            else:
                x = eqx.nn.Sequential(self.body_repvits_no_pad)(x)
        return x

    def __call__(self, x, state, key=None, get_intermediates=False):
        if get_intermediates:
            head = self.head(x)
            body, body_intermediates = self.body(head, get_intermediates=True)
            body += head[:, self.body_spatial_reduction:head.shape[1]-self.body_spatial_reduction, self.body_spatial_reduction:head.shape[2]-self.body_spatial_reduction]
            h = self.tail(body)
            base = self.upscale(x)
            out = h + base
            return out, state, [head, *body_intermediates, body]
        else:
            head = self.head(x)
            body = self.body(head) + head[:, self.body_spatial_reduction:head.shape[1]-self.body_spatial_reduction, self.body_spatial_reduction:head.shape[2]-self.body_spatial_reduction]
            h = self.tail(body)
            base = self.upscale(x)
            out = h + base
            return out, state
