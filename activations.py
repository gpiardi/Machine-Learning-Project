"""
Activation Functions and Their Derivatives

Every function is element-wise (except softmax, which works row by row) and
fully vectorized: it accepts a NumPy array of any shape and returns an array
of the same shape.

The derivatives are written as functions of the *pre-activation* z, because
that is what backpropagation stores from the forward pass (z1 in the models).
"""

import numpy as np
from numpy.typing import NDArray

# exp(709) is the largest value that fits in float64. Clipping the input of
# exp() to +/-500 keeps us safely away from overflow, and the sigmoid is
# already 0 or 1 to machine precision long before |z| = 500.
_LOGISTIC_CLIP = 500.0


def identity(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Identity (linear) activation: f(z) = z.

    Used as the output activation of the regressor.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        The input, unchanged
    """
    return z


def identity_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of the identity activation: f'(z) = 1.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Array of ones with the same shape as z
    """
    return np.ones_like(z, dtype=float)


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation function: f(z) = max(0, z).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return np.maximum(0.0, z)


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU activation: f'(z) = 1 if z > 0 else 0.

    ReLU is not differentiable at z = 0; we use the common convention
    f'(0) = 0 (same choice as scikit-learn).

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return (z > 0).astype(float)


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent activation function: f(z) = (e^z - e^-z) / (e^z + e^-z).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values in (-1, 1)
    """
    return np.tanh(z)


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh activation: f'(z) = 1 - tanh(z)^2.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return 1.0 - np.tanh(z) ** 2


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation function: f(z) = 1 / (1 + e^-z).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values in (0, 1)
    """
    # Clip so that np.exp(-z) cannot overflow for very negative z.
    z = np.clip(z, -_LOGISTIC_CLIP, _LOGISTIC_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of logistic activation: f'(z) = f(z) * (1 - f(z)).

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values in (0, 0.25]
    """
    s = logistic(z)
    return s * (1.0 - s)


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer (classification).

    softmax(z)_k = exp(z_k) / sum_j exp(z_j), computed independently for
    every row (sample).

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    # Softmax is invariant to adding a constant to every logit of a row, so
    # subtracting the row maximum changes nothing mathematically but makes the
    # largest exponent exp(0) = 1, which rules out overflow (e.g. z = 1000).
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
