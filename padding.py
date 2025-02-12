import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import scipy
import sympy
from typing import Literal
from scipy.special import binom

# Generate weights for Lagrange polynomial extrapolation, using an empirically derived recursion.
# The generated weights match those solved by:
# https://github.com/hamk-uas/DifferentialConv2d/blob/main/diff_conv2d/lagrange_constants/gen_lagrange_padding_coefs.py
def make_extr_pad_weights(max_num_padding, num_predictors):
    """Generate weights for Lagrange polynomial extrapolation, using an empirically derived recursion.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    for k in range(num_predictors):
        if k == 0:
            cumsumconsts = np.array([[1]])
        else:
            b = np.array(binom(k, np.arange(k+1)), dtype=int)
            b = b*np.expand_dims(b, 1)
            b[1:,1:] += cumsumconsts
            cumsumconsts = b
    c = np.tile(cumsumconsts[0], (max(max_num_padding - k, 0), 1))
    for i in range(1, k + 1):
        c = np.vstack([cumsumconsts[i], c])
        if c.shape[0] > max_num_padding:
            c = c[:max_num_padding,:]
        c = np.cumsum(c, axis=0)
    c = c*(-1)**np.flip(np.arange(k+1))
    return jnp.array(c.T)

# Lagrange polynomial padding
def extr_pad(x, num_padding: tuple, num_predictors: int):
    # Automatically reduce num_predictors if x is too small
    num_predictors = (min(x.shape[0], num_predictors), min(x.shape[1], num_predictors))
    weights = (
        make_extr_pad_weights(np.max(np.array(num_padding[0])), num_predictors[0]),
        make_extr_pad_weights(np.max(np.array(num_padding[1])), num_predictors[1])
    )
    left_padding = jnp.matmul(x[:, :num_predictors[1]], jnp.flip(weights[1][:, :num_padding[1][0]], axis=(0, 1)))
    right_padding = jnp.matmul(x[:, -num_predictors[1]:], weights[1][:, :num_padding[1][1]])
    x = jnp.concatenate((left_padding, x, right_padding), axis = 1)
    top_padding = jnp.matmul(x[:num_predictors[0], :].T, jnp.flip(weights[0][:, :num_padding[0][0]], axis=(0, 1))).T
    bottom_padding = jnp.matmul(x[-num_predictors[0]:, :].T, weights[0][:, :num_padding[0][1]]).T
    x = jnp.concatenate((top_padding, x, bottom_padding), axis = 0)
    return x

# Lagrange polynomial padding, vmapped over channels
def extr_pad_channels(data, num_padding, **kwargs):
    return jax.vmap(lambda data: extr_pad(data, num_padding, **kwargs), axis_name="channel")(data)


# Test whether integer n only includes factors in iterable allowed_factors
def only_has_factors(n, allowed_factors):
    factors = sympy.factorint(n)
    for factor in allowed_factors:
        if factor in factors:
            del factors[factor]
    return len(factors.keys()) == 0

# Get smallest FFT size >= n that only has CUDA preferred integer factors
def get_fft_size(n):
    while not only_has_factors(n, [2, 3, 5, 7]):
        n += 1
    return n


# Multiply data by a window function
def apodize(data, apodization):
    if apodization == "tukey":
        # Tukey window apodization
        window_x = np.array(scipy.signal.windows.tukey(data.shape[1] + 1, sym=False)[1:])
        window_y = np.array([scipy.signal.windows.tukey(data.shape[0] + 1, sym=False)[1:]]).T
        anal_data = data*window_y*window_x
    elif apodization is None:
        # No apodization
        anal_data = data
    return anal_data


# Get a system from which linear prediction coefficients can be solved, based on data autocorrelation.
# anal_data: data to analyze, already windowed
# length: length of the neighborhood in the padding direction
# width: width of the neighborhood perpendicular to the padding direction
# correlate_method: "fft" or "direct"
def get_system(anal_data: jax.Array, length: int, width: int, correlate_method: str):
    # length, width: shape of the rectangle of predictor pixels when padding downwards
    assert length > 0, "Length must be > 0"
    assert width > 0 and width%2 == 1, "width must be odd and > 0"
    max_offset = max(length + 1, width) - 1  # Largest horizontal or vertical offset for which we need to calculate correlation

    # Analysis
    # Calculate xcor_matrix indexed by shifted offsets (y_offset, x_offset - max_offset)
    if correlate_method == "fft":
        # FFT-based correlation
        # Zero pad to eliminate wraparound across image boundary. Add extra padding as needed to use an optimal FFT size.
        anal_data = jnp.pad(anal_data, ((0, get_fft_size(anal_data.shape[0]+max_offset) - anal_data.shape[0]), (0, get_fft_size(anal_data.shape[1]+max_offset) - anal_data.shape[1])))
        # Real discrete fourier tranform
        anal_data_dft = jnp.fft.rfft2(anal_data, norm="forward")
        # Calculate correlation by frequency domain multiplication by complex conjugate, inverse real DFT, roll and clip to include the range of offsets needed
        xcor_matrix = jnp.roll(jnp.fft.irfft2(anal_data_dft*jnp.conj(anal_data_dft), norm="forward"), max_offset, axis=1)[:max_offset+1, :max_offset+max_offset+1]
    elif correlate_method == "direct":
        # Direct correlation
        # Zero pad to enable covariance calculation at non-zero offsets.
        padded_anal_data = jnp.pad(anal_data, ((0, max_offset), (max_offset, max_offset)))
        def cross_correl(shifted_offset):
            return jnp.mean(anal_data*jax.lax.dynamic_slice(padded_anal_data, shifted_offset, anal_data.shape))
        vmapped_cross_correl = jax.vmap(cross_correl)
        # non-negative indices to xcor_matrix
        shifted_offset_to_shifted_offset = np.stack(
            np.broadcast_arrays(np.expand_dims(np.arange(max_offset+1), -1), np.arange(max_offset*2+1)), axis=-1)  # [y, x, axis]
        # Could do this:
        # xcor_matrix = jax.vmap(vmapped_cross_correl)(yx_to_yx)
        # But instead avoid calculating unused elements at xcor_matrix[0,:max_offset]:
        xcor_matrix = jnp.block([
            [jnp.zeros((max_offset,)), vmapped_cross_correl(shifted_offset_to_shifted_offset[0,max_offset:])],
            [jax.vmap(vmapped_cross_correl)(shifted_offset_to_shifted_offset[1:,:])]
        ])
    else:
        raise Exception(f"Unknown correlate_method ""{correlate_method}""")
    
    # yx are non-negative coordinates within any rotation of the analysis neighborhood
    yx_to_yx = np.stack(np.broadcast_arrays(np.expand_dims(np.arange(max(length + 1, width)), -1), np.arange(max(length + 1, width))), axis=-1)  # [y, x, axis]

    # Signed offsets between (y1, x1) and (y0, x0)
    yx_pairs_to_offsets = np.expand_dims(yx_to_yx, axis = (2, 3)) - yx_to_yx  # [y1, x1, y0, x0, axis]

    # Shifted non-negative offsets by which xcor_matrix can be addressed
    yx_pairs_to_cov_index = np.apply_along_axis(
        lambda yx: np.array([-yx[0], max_offset-yx[1]]) if yx[0] < 0 or (yx[0] == 0 and yx[1] < 0) else np.array([yx[0], max_offset+yx[1]]), -1, yx_pairs_to_offsets) # [y1, x1, y0, x0, axis]

    # Get indices to xcor_matrix for obtaining covariances between flattened xys in two rectangular areas, with optional swapping of flattening order
    def yx_ranges_to_xcor_matrix_indices(yx_ranges_0, yx_ranges_1, flatten_swap_xy):
        yx_to_yx_0 = np.stack(np.broadcast_arrays(np.expand_dims(yx_ranges_0[0], -1), yx_ranges_0[1]), axis=-1)  # [y, x, axis]
        yx_to_yx_1 = np.stack(np.broadcast_arrays(np.expand_dims(yx_ranges_1[0], -1), yx_ranges_1[1]), axis=-1)  # [y, x, axis]
        if flatten_swap_xy:
            # Swap flattening order
            yx_to_yx_0 = np.swapaxes(yx_to_yx_0, 0, 1)  # [x, y, axis]
            yx_to_yx_1 = np.swapaxes(yx_to_yx_1, 0, 1)  # [x, y, axis]
        yxs0 = np.concatenate(yx_to_yx_0)  # [y x flattened, axis]
        yxs1 = np.concatenate(yx_to_yx_1)  # [y x flattened, axis]
        yx_pairs_to_yx_pairs = np.stack(np.broadcast_arrays(np.expand_dims(yxs0, 1), yxs1), axis=-2)  # [y0 x0 flattened, y1 x1 flattened, axis_0, axis_1]
        return yx_pairs_to_cov_index[yx_pairs_to_yx_pairs[..., 0, 0], yx_pairs_to_yx_pairs[..., 0, 1], yx_pairs_to_yx_pairs[..., 1, 0], yx_pairs_to_yx_pairs[..., 1, 1]]  # [y0 x0 flattened, y1 x1 flattened, axis]

    # Ranges of ys and xs in rectangular areas used as predictors and predicted
    down_predictor_yx_ranges = (np.arange(length), np.arange(width))  # (range of y, range of x)
    down_predicted_yx_ranges = (np.arange(length, length + 1), np.arange(width))  # (range of y, range of x)
    right_predictor_yx_ranges = (np.arange(width), np.arange(length))  # (range of y, range of x)
    right_predicted_yx_ranges = (np.arange(width), np.arange(length, length + 1))  # (range of y, range of x)

    # Fill systems_matrix and systems_vector from xcor_matrix
    down_system_matrix_xcor_matrix_indices = yx_ranges_to_xcor_matrix_indices(down_predictor_yx_ranges, down_predictor_yx_ranges, False)  # [y x flattened, y x flattened, axis]
    down_system_vector_xcor_matrix_indices = yx_ranges_to_xcor_matrix_indices(down_predicted_yx_ranges, down_predictor_yx_ranges, False)  # [y x flattened, y x flattened, axis]
    right_system_matrix_xcor_matrix_indices = yx_ranges_to_xcor_matrix_indices(right_predictor_yx_ranges, right_predictor_yx_ranges, True)  # [y x flattened, y x flattened, axis]
    right_system_vector_xcor_matrix_indices = yx_ranges_to_xcor_matrix_indices(right_predicted_yx_ranges, right_predictor_yx_ranges, True)  # [y x flattened, y x flattened, axis]
    systems_matrix_xcor_matrix_indices = np.stack((down_system_matrix_xcor_matrix_indices, right_system_matrix_xcor_matrix_indices)) # [padding axis, y x flattened, y x flattened, axis]
    systems_vector_xcor_matrix_indices = np.stack((down_system_vector_xcor_matrix_indices, right_system_vector_xcor_matrix_indices)) # [padding axis, y x flattened, y x flattened, axis]
    systems_matrix = xcor_matrix[systems_matrix_xcor_matrix_indices[..., 0], systems_matrix_xcor_matrix_indices[..., 1]]  # [padding axis, y x flattened, y x flattened, axis]
    systems_vector = xcor_matrix[systems_vector_xcor_matrix_indices[..., 0], systems_vector_xcor_matrix_indices[..., 1]]  # [padding axis, y x flattened, y x flattened, axis]

    return systems_matrix, systems_vector


# Solve coefficients from the given system
def solve_coefs(systems_matrix, systems_vector, length, width, cholesky_stab):
    def solve(system_matrix, system_vector):
        operator = lx.MatrixLinearOperator(system_matrix + np.eye(system_matrix.shape[0])*cholesky_stab, lx.positive_semidefinite_tag)
        coefs = jnp.ravel(lx.linear_solve(operator, system_vector, solver=lx.Cholesky(), throw=False).value)
        coefs = jnp.nan_to_num(coefs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return coefs
    coefs = jax.vmap(
        jax.vmap(
            solve,
            axis_name="predicted_y", in_axes=(None, 0), out_axes=0
        ),
        axis_name="num_rotation"
    )(systems_matrix, systems_vector)
    return jnp.reshape(coefs, (2, width, length, width))  # [direction, prediction x, predictor y, predictor x]


# Function for scan, to predict downwards
def predict_down(carry, coefs):
    if coefs.shape[0] > 1:
        bottom_left_pred = jnp.sum(coefs[:coefs.shape[0]//2] * carry[:,:coefs.shape[0]], axis=(1,2))
    def pred_bottom_mid(x):
        return jnp.sum(coefs[coefs.shape[0]//2]*jax.lax.dynamic_slice_in_dim(carry, x, coefs.shape[0], axis=1))
    bottom_mid_pred = jax.vmap(pred_bottom_mid)(np.arange(carry.shape[1] + 1 - coefs.shape[0]))
    if coefs.shape[0] > 1:
        bottom_right_pred = jnp.sum(coefs[-(coefs.shape[0]//2):] * carry[:,-coefs.shape[0]:], axis=(1,2))
        predicted_values = jnp.concat((bottom_left_pred, bottom_mid_pred, bottom_right_pred))
    else:
        predicted_values = bottom_mid_pred
    # move one column ahead by removing leftmost col and adding predicted values to the right
    carry = jnp.concat((carry[1:], jnp.expand_dims(predicted_values, axis=0)), axis=0)
    return carry, predicted_values


# Linear prediction pad data
def pad(data, num_padding, coefs):
    # coefs should be in order [bottom, right, top, left] or [shared bottom top, shared right left] 1D-ordered from farthest to nearest predictor
    
    # Pad horizontally
    pred_left = np.flip(jax.lax.scan(lambda carry, xs: predict_down(carry, coefs[1] if len(coefs) == 2 else coefs[3]), init=jnp.flip(data[:,:coefs.shape[2]].T), xs=None, length=num_padding[1][0])[1]).T
    pred_right = jax.lax.scan(lambda carry, xs: predict_down(carry, coefs[1]), init=data[:,-coefs.shape[2]:].T, xs=None, length=num_padding[1][1])[1].T
    data = jnp.concat((pred_left, data, pred_right), axis=1)

    # Pad vertically
    pred_top = np.flip(jax.lax.scan(lambda carry, xs: predict_down(carry, coefs[0] if len(coefs) == 2 else coefs[2]), init=jnp.flip(data[:coefs.shape[2],:]), xs=None, length=num_padding[0][0])[1])
    pred_bottom = jax.lax.scan(lambda carry, xs: predict_down(carry, coefs[0]), init=data[-coefs.shape[2]:,:], xs=None, length=num_padding[0][1])[1]
    data =  jnp.concat((pred_top, data, pred_bottom), axis=0)

    return data

# Safe division and its custom derivative
@jax.custom_vjp
def safe_divide(a, b):
    div = a/b
    return jnp.where(jnp.isfinite(div), div, 0)

def safe_divide_fwd(a, b):
    return safe_divide(a, b), (a, b)

def safe_divide_bwd(res, g):
    (a, b) = res
    div_a = g/b
    div_b = -g*a/b**2
    valid = jnp.logical_and(jnp.isfinite(div_a), jnp.isfinite(div_b))
    return (jnp.where(valid, div_a, 0), jnp.where(valid, div_b, 0))
    # This version would enable higher-order differentiation but we did not use it during experiments:
    # return safe_divide(g, b), safe_divide(-g*a, b**2)

safe_divide.defvjp(safe_divide_fwd, safe_divide_bwd)


# Reciprocate the complex conjugate pair of poles if they are are outside the unit circle
def stabilize_2nd_order_predictor_with_complex_poles(coefs):
    return jax.lax.select(coefs[1] < -1.0, jnp.array([-coefs[0]/coefs[1], 1.0/coefs[1]]), coefs)

def stabilize_2nd_order_predictor_with_real_poles(coefs):
    # Reciprocate each real pole if it is outside the unit circle
    # Find poles
    s = jnp.sqrt(coefs[0]**2 + 4.0*coefs[1])*0.5
    poles = coefs[0]*0.5 + jnp.array([s, -s])
    # Reciprocate if needed
    poles = jax.lax.select(jnp.abs(poles) > 1.0, 1.0/poles, poles)
    # Return coefficients
    return jnp.array([poles[0] + poles[1], -poles[0]*poles[1]])

def stabilize_1st_order_predictor(a_1):
    # Reciprocate the real pole if it is outside the unit circle
    return jax.lax.select(jnp.abs(a_1) > 1.0, 1.0/a_1, a_1)

# Stabilize a 2nd order recurrent 1-d predictor
def stabilize_2nd_order_predictor(coefs):
    # Check if we have a complex conjugate pair of poles. Otherwise we have a pair of real poles. Handle the cases differently
    return jax.lax.select(coefs[0]*coefs[0] + 4.0*coefs[1] < 0.0, stabilize_2nd_order_predictor_with_complex_poles(coefs), stabilize_2nd_order_predictor_with_real_poles(coefs))

# Gather covariance method statistics
def get_covariances(anal_data, correlate_method, length, width):
    # Covariance method and symmetric AR process covariance method implemented as special case for 2x1 and 1x1 neighborhoods
    assert width == 1 and (length == 1 or length == 2), f'correlate_method = "{correlate_method}" has only been implemented for width = 1, length = 1 or 2'

    reuse_partial_sums = True  # Use hand-optimized reuse of partial sums?
    if reuse_partial_sums:
        if length == 1:
            center = jnp.sum(anal_data[1:-1,1:-1]*anal_data[1:-1,1:-1])
            top = jnp.sum(anal_data[0,1:-1]*anal_data[0,1:-1])
            bottom = jnp.sum(anal_data[-1,1:-1]*anal_data[-1,1:-1])
            left = jnp.sum(anal_data[1:-1,0]*anal_data[1:-1,0])
            right = jnp.sum(anal_data[1:-1,-1]*anal_data[1:-1,-1])
            top_left = anal_data[0,0]*anal_data[0,0]
            top_right = anal_data[0,-1]*anal_data[0,-1]
            bottom_left = anal_data[-1,0]*anal_data[-1,0]
            bottom_right = anal_data[-1,-1]*anal_data[-1,-1]

            r_11 = top_left + top + left + center + bottom_left + bottom
            l_11 = top + top_right + center + right + bottom + bottom_right
            b_11 = top_left + top + top_right + left + center + right
            t_11 = left + center + right + bottom_left + bottom + bottom_right

            r_01 = jnp.sum(anal_data[:,1:]*anal_data[:,:-1])
            l_01 = r_01

            b_01 = jnp.sum(anal_data[1:,:]*anal_data[:-1,:])
            t_01 = b_01

        if length == 2:
            m = jnp.sum(anal_data[2:-2, 2:-2]*anal_data[2:-2, 2:-2])
            l2 = jnp.sum(anal_data[2:-2, 0]*anal_data[2:-2, 0])
            l1 = jnp.sum(anal_data[2:-2, 1]*anal_data[2:-2, 1])
            r1 = jnp.sum(anal_data[2:-2, -2]*anal_data[2:-2, -2])
            r2 = jnp.sum(anal_data[2:-2, -1]*anal_data[2:-2, -1])
            t2 = jnp.sum(anal_data[0, 2:-2]*anal_data[0, 2:-2])
            t1 = jnp.sum(anal_data[1, 2:-2]*anal_data[1, 2:-2])
            b1 = jnp.sum(anal_data[-2, 2:-2]*anal_data[-2, 2:-2])
            b2 = jnp.sum(anal_data[-1, 2:-2]*anal_data[-1:, 2:-2])

            h0 = jnp.sum(anal_data[0:2, 0]*anal_data[0:2, 0]) + l2 + jnp.sum(anal_data[-2:, 0]*anal_data[-2:, 0])
            h1 = jnp.sum(anal_data[0:2, 1]*anal_data[0:2, 1]) + l1 + jnp.sum(anal_data[-2:, 1]*anal_data[-2:, 1])
            h = t2 + t1 + m + b1 + b2
            hm2 = jnp.sum(anal_data[0:2, -2]*anal_data[0:2, -2]) + r1 + jnp.sum(anal_data[-2:, -2]*anal_data[-2:, -2])
            hm1 = jnp.sum(anal_data[0:2, -1]*anal_data[0:2, -1]) + r2 + jnp.sum(anal_data[-2:, -1]*anal_data[-2:, -1])

            r_11 = h1 + h + hm2
            l_11 = r_11
            r_22 = h0 + h1 + h
            l_22 = h + hm2 + hm1

            v0 = jnp.sum(anal_data[0, 0:2]*anal_data[0, 0:2]) + t2 + jnp.sum(anal_data[0, -2:]*anal_data[0, -2:])
            v1 = jnp.sum(anal_data[1, 0:2]*anal_data[1, 0:2]) + t1 + jnp.sum(anal_data[1, -2:]*anal_data[1, -2:])
            v = l2 + l1 + m + r1 + r2
            vm2 = jnp.sum(anal_data[-2, 0:2]*anal_data[-2, 0:2]) + b1 + jnp.sum(anal_data[-2, -2:]*anal_data[-2, -2:])
            vm1 = jnp.sum(anal_data[-1, 0:2]*anal_data[-1, 0:2]) + b2 + jnp.sum(anal_data[-1, -2:]*anal_data[-1, -2:])

            b_11 = v1 + v + vm2
            t_11 = b_11
            b_22 = v0 + v1 + v
            t_22 = v + vm2 + vm1

            hm1 = jnp.sum(anal_data[:,1:-2]*anal_data[:,2:-1])
            hl1 = jnp.sum(anal_data[:,0]*anal_data[:,1])
            hr1 = jnp.sum(anal_data[:,-2]*anal_data[:,-1])

            r_12 = hl1 + hm1
            r_01 = hm1 + hr1
            l_01 = r_12
            l_12 = r_01
            
            h2 = jnp.sum(anal_data[:,:-2]*anal_data[:,2:])
            l_02 = h2
            r_02 = h2

            vm1 = jnp.sum(anal_data[1:-2,:]*anal_data[2:-1,:])
            vt1 = jnp.sum(anal_data[0,:]*anal_data[1,:])
            vb1 = jnp.sum(anal_data[-2,:]*anal_data[-1,:])

            b_12 = vt1 + vm1
            b_01 = vm1 + vb1
            t_01 = b_12
            t_12 = b_01
            
            v2 = jnp.sum(anal_data[:-2,:]*anal_data[2:,:])
            t_02 = v2
            b_02 = v2
    else:
        if length == 1:
            r_11 = jnp.sum(anal_data[:, :,:-1]*anal_data[:,:-1])
            r_01 = jnp.sum(anal_data[:, :,1:]*anal_data[:,:-1])

            l_11 = jnp.sum(anal_data[:,1:]*anal_data[:,1:])
            l_01 = jnp.sum(anal_data[:,:-1]*anal_data[:,1:])

            b_11 = jnp.sum(anal_data[:-1,:]*anal_data[:-1,:])
            b_01 = jnp.sum(anal_data[1:,:]*anal_data[:-1,:])

            t_11 = jnp.sum(anal_data[1:,:]*anal_data[1:,:])
            t_01 = jnp.sum(anal_data[:-1,:]*anal_data[1:,:])
        elif length == 2:
            r_11 = jnp.sum(anal_data[:,1:-1]*anal_data[:,1:-1])
            r_22 = jnp.sum(anal_data[:,0:-2]*anal_data[:,0:-2])
            r_01 = jnp.sum(anal_data[:,2:]*anal_data[:,1:-1])
            r_12 = jnp.sum(anal_data[:,1:-1]*anal_data[:,0:-2])
            r_02 = jnp.sum(anal_data[:,2:]*anal_data[:,0:-2])

            l_11 = jnp.sum(anal_data[:,1:-1]*anal_data[:,1:-1])
            l_22 = jnp.sum(anal_data[:,2:]*anal_data[:,2:])
            l_01 = jnp.sum(anal_data[:,:-2]*anal_data[:,1:-1])
            l_12 = jnp.sum(anal_data[:,1:-1]*anal_data[:,2:])
            l_02 = jnp.sum(anal_data[:,:-2]*anal_data[:,2:])

            b_11 = jnp.sum(anal_data[1:-1,:]*anal_data[1:-1,:])
            b_22 = jnp.sum(anal_data[0:-2,:]*anal_data[0:-2,:])
            b_01 = jnp.sum(anal_data[2:,:]*anal_data[1:-1,:])
            b_12 = jnp.sum(anal_data[1:-1,:]*anal_data[0:-2,:])
            b_02 = jnp.sum(anal_data[2:,:]*anal_data[0:-2,:])

            t_11 = jnp.sum(anal_data[1:-1,:]*anal_data[1:-1,:])
            t_22 = jnp.sum(anal_data[2:,:]*anal_data[2:,:])
            t_01 = jnp.sum(anal_data[:-2,:]*anal_data[1:-1,:])
            t_12 = jnp.sum(anal_data[1:-1,:]*anal_data[2:,:])
            t_02 = jnp.sum(anal_data[:-2,:]*anal_data[2:,:])

    if length == 1:
        return r_11, r_01, l_11, l_01, b_11, b_01, t_11, t_01
    elif length == 2:
        return r_11, r_22, r_01, r_12, r_02, l_11, l_22, l_01, l_12, l_02, b_11, b_22, b_01, b_12, b_02, t_11, t_22, t_01, t_12, t_02
    
# Gather covariance/autocorrelation statistics and solve normal equations to obtain coefficients
def solve_coefs_cov(anal_data, correlate_method, length, width):
    # Covariance method and symmetric AR process covariance method implemented as special case for 2x1 and 1x1 neighborhoods
    assert width == 1 and (length == 1 or length == 2), f'correlate_method = "{correlate_method}" has only been implemented for width = 1, length = 1 or 2'
    
    if length == 1:
        r_11, r_01, l_11, l_01, b_11, b_01, t_11, t_01 = get_covariances(anal_data, correlate_method, length, width)
    elif length == 2:
        r_11, r_22, r_01, r_12, r_02, l_11, l_22, l_01, l_12, l_02, b_11, b_22, b_01, b_12, b_02, t_11, t_22, t_01, t_12, t_02 = get_covariances(anal_data, correlate_method, length, width)

    if correlate_method == "cov" or correlate_method == "cov_stab":
        if length == 1:
            ra_1 = safe_divide(r_01, r_11)
            la_1 = safe_divide(l_01, l_11)
            ba_1 = safe_divide(b_01, b_11)
            ta_1 = safe_divide(t_01, t_11)
            if correlate_method == "cov_stab":
                ra_1 = stabilize_1st_order_predictor(ra_1)
                la_1 = stabilize_1st_order_predictor(la_1)
                ba_1 = stabilize_1st_order_predictor(ba_1)
                ta_1 = stabilize_1st_order_predictor(ta_1)
        elif length == 2:
            ra_1 = safe_divide(r_01*r_22 - r_02*r_12, r_11*r_22 - r_12*r_12)
            ra_2 = safe_divide(r_02*r_11 - r_01*r_12, r_11*r_22 - r_12*r_12)

            la_1 = safe_divide(l_01*l_22 - l_02*l_12, l_11*l_22 - l_12*l_12)
            la_2 = safe_divide(l_02*l_11 - l_01*l_12, l_11*l_22 - l_12*l_12)

            ba_1 = safe_divide(b_01*b_22 - b_02*b_12, b_11*b_22 - b_12*b_12)
            ba_2 = safe_divide(b_02*b_11 - b_01*b_12, b_11*b_22 - b_12*b_12)

            ta_1 = safe_divide(t_01*t_22 - t_02*t_12, t_11*t_22 - t_12*t_12)
            ta_2 = safe_divide(t_02*t_11 - t_01*t_12, t_11*t_22 - t_12*t_12)

            if correlate_method == "cov_stab":
                r_coefs = stabilize_2nd_order_predictor(jnp.array([ra_1, ra_2]))
                ra_1, ra_2 = (r_coefs[0], r_coefs[1])
                l_coefs = stabilize_2nd_order_predictor(jnp.array([la_1, la_2]))
                la_1, la_2 = (l_coefs[0], l_coefs[1])
                b_coefs = stabilize_2nd_order_predictor(jnp.array([ba_1, ba_2]))
                ba_1, ba_2 = (b_coefs[0], b_coefs[1])
                t_coefs = stabilize_2nd_order_predictor(jnp.array([ta_1, ta_2]))
                ta_1, ta_2 = (t_coefs[0], t_coefs[1])

    elif correlate_method == "cov_sym" or correlate_method == "cov_sym_stab":
        if length == 1:
            h_01 = r_01 + l_01
            h_11 = r_11 + l_11
            v_01 = b_01 + t_01
            v_11 = b_11 + t_11
            ha_1 = safe_divide(h_01, h_11)
            va_1 = safe_divide(v_01, v_11)
            if correlate_method == "cov_stab":
                ha_1 = stabilize_1st_order_predictor(ha_1)
                va_1 = stabilize_1st_order_predictor(va_1)
        elif length == 2:
            h_11 = r_11 + l_11
            h_22 = r_22 + l_22
            h_01 = r_01 + l_01
            h_12 = r_12 + l_12
            h_02 = r_02 + l_02

            ha_1 = safe_divide(h_01*h_22 - h_02*h_12, h_11*h_22 - h_12*h_12)
            ha_2 = safe_divide(h_02*h_11 - h_01*h_12, h_11*h_22 - h_12*h_12)

            v_11 = b_11 + t_11
            v_22 = b_22 + t_22
            v_01 = b_01 + t_01
            v_12 = b_12 + t_12
            v_02 = b_02 + t_02

            va_1 = safe_divide(v_01*v_22 - v_02*v_12, v_11*v_22 - v_12*v_12)
            va_2 = safe_divide(v_02*v_11 - v_01*v_12, v_11*v_22 - v_12*v_12)

            if correlate_method == "cov_sym_stab":
                h_coefs = stabilize_2nd_order_predictor(jnp.array([ha_1, ha_2]))
                ha_1, ha_2 = (h_coefs[0], h_coefs[1])
                v_coefs = stabilize_2nd_order_predictor(jnp.array([va_1, va_2]))
                va_1, va_2 = (v_coefs[0], v_coefs[1])

    if correlate_method == "cov" or correlate_method == "cov_stab":
        if length == 1:
            coefs = jnp.array([
                [[[ba_1]]],  # bottom
                [[[ra_1]]],  # right
                [[[ta_1]]],  # top
                [[[la_1]]]  # left
            ])
        elif length == 2:
            coefs = jnp.array([
                [[[ba_2], [ba_1]]],  # bottom
                [[[ra_2], [ra_1]]],  # right
                [[[ta_2], [ta_1]]],  # top
                [[[la_2], [la_1]]]  # left
            ])
    elif correlate_method == "cov_sym" or correlate_method == "cov_sym_stab":
        if length == 1:
            coefs = jnp.array([
                [[[va_1]]],  # vertical
                [[[ha_1]]]  # horizontal
            ])
        elif length == 2:
            coefs = jnp.array([
                [[[va_2], [va_1]]],  # vertical
                [[[ha_2], [ha_1]]]  # horizontal
            ])
    return coefs

# Pad 2D (y,x) data using linear prediction padding
def linear_prediction_pad(
        data: jax.Array,
        num_padding: tuple,
        length: int=2,
        width: int=3,
        apodization: Literal["tukey", None] = "tukey",
        correlate_method: Literal["direct", "fft", "cov", "cov_sym", "cov_stab", "cov_sym_stab"] = "fft",
        cholesky_stab: float=1e-7
    ):
    # Subtract mean
    mean = jnp.mean(data)
    data = data - mean

    # Apodize
    anal_data = apodize(data, apodization)

    if correlate_method == "cov" or correlate_method == "cov_sym" or correlate_method == "cov_stab" or correlate_method == "cov_sym_stab":
        # Use covariance method, optionally assuming symmetric AR process
        coefs = solve_coefs_cov(anal_data, correlate_method, length, width)
    else:
        # Analyze data to obtain the system
        systems_matrix, systems_vector = get_system(anal_data, length, width, correlate_method)

        # Solve the system to obtain linear prediction coefficients
        coefs = solve_coefs(systems_matrix, systems_vector, length, width, cholesky_stab)

    # Pad data
    data = pad(data, num_padding, coefs)

    # Restore mean
    data = data + mean
    
    return data

# Pad channel,y,x data using linear prediction padding
def linear_prediction_pad_channels(data, num_padding, **kwargs):
    return jax.vmap(lambda data: linear_prediction_pad(data, num_padding, **kwargs), axis_name="channel")(data)

# An Equinox module that can be used as a configurable padding layer
class Padding2dLayer(eqx.Module):
    num_padding: tuple = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    padding_method_kwargs: dict = eqx.field(static=True)

    def __init__(self,
        num_padding,
        mode: Literal["lp", "zero", "repl", "extr"],
        padding_method_kwargs: dict = {},
        
    ):
        self.num_padding = num_padding
        self.mode = mode
        self.padding_method_kwargs = padding_method_kwargs

    def __call__(self, x, key=None):
        if self.mode == "lp":
            return linear_prediction_pad_channels(x, self.num_padding, **self.padding_method_kwargs)
        elif self.mode == "extr":
            return extr_pad_channels(x, self.num_padding, **self.padding_method_kwargs)
        else:
            num_padding = ((0,0), ) + self.num_padding  # no padding for channel dimension 
            if self.mode == "zero":
                return jnp.pad(x, num_padding, mode="constant", constant_values=0.0)
            elif self.mode == "repl":
                return jnp.pad(x, num_padding, mode="edge")
            else:
                raise Exception(f"Unknown mode \"{self.mode}\"")