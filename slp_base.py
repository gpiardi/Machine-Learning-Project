"""
Base class for SLP Classifier and Regressor
"""

import inspect
import numbers
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from activations import (
    identity,
    identity_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)

# Hidden-layer activations: name -> (function, derivative w.r.t. z)
_ACTIVATIONS = {
    "identity": (identity, identity_derivative),
    "logistic": (logistic, logistic_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}
_SOLVERS = ("sgd", "adam")

# Adam hyper-parameters, fixed to the values recommended by Kingma & Ba (2015)
# (also scikit-learn's defaults).
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPSILON = 1e-8

# Mini-batch size used when batch_size="auto" (same rule as scikit-learn).
_AUTO_BATCH_SIZE = 200

Parameters = Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]


class ConvergenceWarning(UserWarning):
    """Training reached max_iter before the stopping criterion was met."""


class NotFittedError(ValueError, AttributeError):
    """predict/score was called before fit."""


def _check_integer(name: str, value: Any, minimum: int) -> None:
    """Raise ValueError unless value is an integer >= minimum (bools excluded)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _check_real(
    name: str,
    value: Any,
    low: float,
    high: float = np.inf,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> None:
    """Raise ValueError unless value is a finite real number inside the interval."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    above_low = value >= low if low_inclusive else value > low
    below_high = value <= high if high_inclusive else value < high
    if not (np.isfinite(value) and above_low and below_high):
        left = "[" if low_inclusive else "("
        right = "]" if high_inclusive else ")"
        raise ValueError(
            f"{name} must be in {left}{low}, {high}{right}, got {value!r}"
        )


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators:
    hyper-parameter and input validation, weight initialization, the
    mini-batch training loop, the optimizers (SGD with momentum and Adam),
    the hidden-layer part of backpropagation and early stopping.

    Subclasses only define what differs between the two tasks: how targets
    are encoded, the output activation, the loss and its gradient with
    respect to the output pre-activation z2, and predict/score.

    Notation (n = n_samples, d = n_features, h = hidden_layer_size,
    k = n_outputs):
        X  : (n, d)    W1 : (d, h)    b1 : (h,)
        z1 : (n, h)    a1 : (n, h)    W2 : (h, k)    b2 : (k,)
        z2 : (n, k)    y_pred : (n, k)
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        solver: str = "adam",
        momentum: float = 0.9,
        batch_size: Union[int, str] = "auto",
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
    ) -> None:
        """
        Initialize the SLP estimator.

        Following the scikit-learn convention, __init__ only stores the
        hyper-parameters; they are validated when fit() is called.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations (epochs, i.e. full passes over the
            training data)
        random_state : int or None
            Random seed for reproducibility (weight initialization, shuffling
            of mini-batches and the early-stopping validation split)
        solver : {'sgd', 'adam'}, default='adam'
            'sgd' is (mini-batch) stochastic gradient descent with optional
            momentum; 'adam' is the adaptive Adam optimizer
        momentum : float in [0, 1), default=0.9
            Momentum for the 'sgd' solver (0 gives plain SGD). Ignored by Adam.
        batch_size : int or 'auto', default='auto'
            Size of the mini-batches. 'auto' uses min(200, n_samples);
            a value >= n_samples gives full-batch gradient descent and 1 gives
            pure (one sample at a time) stochastic gradient descent.
        early_stopping : bool, default=False
            If True, hold out validation_fraction of the training data and
            stop when the validation loss has not improved by at least tol
            for n_iter_no_change epochs; the best weights are restored.
            If False, the same rule is applied to the training loss.
        validation_fraction : float in (0, 1), default=0.1
            Fraction of the training data used for early stopping
        n_iter_no_change : int, default=10
            Number of epochs without improvement before stopping
        tol : float, default=1e-4
            Minimum decrease of the monitored loss that counts as improvement
        """
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.solver: str = solver
        self.momentum: float = momentum
        self.batch_size: Union[int, str] = batch_size
        self.early_stopping: bool = early_stopping
        self.validation_fraction: float = validation_fraction
        self.n_iter_no_change: int = n_iter_no_change
        self.tol: float = tol

        # To be initialized in fit()
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Track loss during training
        self.validation_loss_curve_: list[float] = []  # Only with early_stopping
        self.n_iter_: int = 0  # Number of epochs actually run
        self.n_features_in_: Optional[int] = None
        self.n_outputs_: Optional[int] = None

    # ------------------------------------------------------------------
    # scikit-learn style parameter handling
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the hyper-parameters passed to __init__ (sklearn convention)."""
        names = inspect.signature(type(self).__init__).parameters
        return {name: getattr(self, name) for name in names if name != "self"}

    def set_params(self, **params: Any) -> "BaseSLPEstimator":
        """Set hyper-parameters by name and return self (sklearn convention)."""
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)}"
                )
            setattr(self, name, value)
        return self

    def _validate_hyperparameters(self) -> None:
        """Check every hyper-parameter and raise ValueError on the first bad one."""
        _check_integer("hidden_layer_size", self.hidden_layer_size, minimum=1)
        _check_integer("max_iter", self.max_iter, minimum=1)
        _check_integer("n_iter_no_change", self.n_iter_no_change, minimum=1)
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, "
                f"got {self.activation!r}"
            )
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}")
        _check_real("learning_rate", self.learning_rate, low=0, low_inclusive=False)
        _check_real("momentum", self.momentum, low=0, high=1, high_inclusive=False)
        _check_real(
            "validation_fraction", self.validation_fraction,
            low=0, high=1, low_inclusive=False, high_inclusive=False,
        )
        _check_real("tol", self.tol, low=0)
        if not (isinstance(self.batch_size, str) and self.batch_size == "auto"):
            _check_integer("batch_size", self.batch_size, minimum=1)
        if not isinstance(self.early_stopping, (bool, np.bool_)):
            raise ValueError(
                f"early_stopping must be a bool, got {self.early_stopping!r}"
            )
        if self.random_state is not None:
            _check_integer("random_state", self.random_state, minimum=0)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_X(X: NDArray) -> NDArray[np.floating]:
        """Convert X to a 2D float array and reject empty or non-finite input."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2D array of shape (n_samples, n_features), got "
                f"{X.ndim}D. Use X.reshape(-1, 1) if you have a single feature."
            )
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X must not be empty, got shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        return X

    @staticmethod
    def _check_same_length(X: NDArray, y: NDArray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: "
                f"{X.shape[0]} != {y.shape[0]}"
            )

    def _check_is_fitted(self) -> None:
        if self.W1_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def _validate_X_for_prediction(self, X: NDArray) -> NDArray[np.floating]:
        """Validation shared by predict / predict_proba / score."""
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was "
                f"fitted with {self.n_features_in_} features"
            )
        return X

    # ------------------------------------------------------------------
    # Network building blocks
    # ------------------------------------------------------------------

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        try:
            return _ACTIVATIONS[self.activation]
        except KeyError:
            raise ValueError(f"Unknown activation: {self.activation}") from None

    def _initialize_weights(self) -> None:
        """
        Initialize weights and biases.

        Glorot/Xavier uniform initialization, the scheme used by scikit-learn's
        MLP: every weight and bias of a layer is drawn from U(-r, r) with
        r = sqrt(factor / (fan_in + fan_out)), factor = 6 (or 2 for the
        logistic activation, as in scikit-learn). This keeps the variance of
        the activations, and of the gradients flowing back, roughly the same
        from one layer to the next. Random (rather than zero) biases spread
        the hidden units' boundaries z = 0 across the input space instead of
        making them all pass through the origin.

        We do not use the N(0, 0.01^2) suggested in the scaffold: with such
        tiny weights z1 is ~0 for every sample, so all hidden units start
        almost identical, and the gradient of W1 (proportional to W2 ~ 0.01)
        is tiny, which makes learning slow (see the report for a comparison).

        Requires self._rng, self.n_features_in_ and self.n_outputs_ (set in
        _train before this method is called).
        """
        factor = 2.0 if self.activation == "logistic" else 6.0

        def glorot_uniform(fan_in: int, fan_out: int) -> Tuple[NDArray, NDArray]:
            bound = np.sqrt(factor / (fan_in + fan_out))
            weights = self._rng.uniform(-bound, bound, size=(fan_in, fan_out))
            biases = self._rng.uniform(-bound, bound, size=fan_out)
            return weights, biases

        self.W1_, self.b1_ = glorot_uniform(self.n_features_in_, self.hidden_layer_size)
        self.W2_, self.b2_ = glorot_uniform(self.hidden_layer_size, self.n_outputs_)

    def _gradients_from_output_delta(
        self,
        X: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        delta2: NDArray[np.floating],
    ) -> Parameters:
        """
        Backpropagate the output error to get all four gradients.

        This part of backpropagation is the same for the classifier and the
        regressor; only delta2 = dL/dz2 depends on the loss / output layer.

        Chain rule, with delta2 of shape (n, k) (already divided by n):
            dL/dW2 = a1^T @ delta2                  (h, k)
            dL/db2 = sum of delta2 over samples     (k,)
            delta1 = dL/dz1 = (delta2 @ W2^T) * g'(z1)   (n, h)
            dL/dW1 = X^T @ delta1                   (d, h)
            dL/db1 = sum of delta1 over samples     (h,)
        """
        _, activation_derivative = self._get_activation_function()

        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis=0)

        # Hidden layer error: send delta2 back through W2, then through g'(z1)
        delta1 = (delta2 @ self.W2_.T) * activation_derivative(z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0)

        return dW1, db1, dW2, db2

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def _parameters(self) -> Parameters:
        return self.W1_, self.b1_, self.W2_, self.b2_

    def _initialize_optimizer(self) -> None:
        """Create the optimizer state: one zero array per parameter array."""
        # SGD: velocity. Adam: first moment (mean of gradients).
        self._velocities = [np.zeros_like(p) for p in self._parameters()]
        # Adam only: second moment (mean of squared gradients) and step count.
        self._squared_gradients = [np.zeros_like(p) for p in self._parameters()]
        self._adam_t = 0

    def _update_parameters(self, gradients: Parameters) -> None:
        """
        Apply one optimizer step to W1_, b1_, W2_, b2_ (in place).

        SGD with momentum (momentum = 0 gives plain SGD):
            v <- momentum * v - lr * grad
            w <- w + v
        Adam (Kingma & Ba, 2015):
            m <- b1 * m + (1 - b1) * grad          running mean of gradients
            s <- b2 * s + (1 - b2) * grad^2        running mean of squared grads
            m_hat = m / (1 - b1^t),  s_hat = s / (1 - b2^t)   bias correction
            w <- w - lr * m_hat / (sqrt(s_hat) + eps)
        The loop is over the 4 parameter arrays, not over individual weights:
        each update is a vectorized operation on a whole matrix.
        """
        lr = self.learning_rate
        if self.solver == "sgd":
            for param, grad, velocity in zip(
                self._parameters(), gradients, self._velocities
            ):
                velocity *= self.momentum
                velocity -= lr * grad
                param += velocity
        else:  # adam
            self._adam_t += 1
            correction1 = 1.0 - _ADAM_BETA1 ** self._adam_t
            correction2 = 1.0 - _ADAM_BETA2 ** self._adam_t
            for param, grad, m, s in zip(
                self._parameters(), gradients,
                self._velocities, self._squared_gradients,
            ):
                m *= _ADAM_BETA1
                m += (1.0 - _ADAM_BETA1) * grad
                s *= _ADAM_BETA2
                s += (1.0 - _ADAM_BETA2) * grad**2
                m_hat = m / correction1
                s_hat = s / correction2
                param -= lr * m_hat / (np.sqrt(s_hat) + _ADAM_EPSILON)

    # ------------------------------------------------------------------
    # Training loop (shared by classifier and regressor)
    # ------------------------------------------------------------------

    def _resolve_batch_size(self, n_samples: int) -> int:
        if self.batch_size == "auto":
            return min(_AUTO_BATCH_SIZE, n_samples)
        return min(self.batch_size, n_samples)

    def _train_validation_split(
        self, X: NDArray[np.floating], Y: NDArray[np.floating]
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Randomly hold out validation_fraction of the samples."""
        n_samples = X.shape[0]
        n_validation = int(np.ceil(self.validation_fraction * n_samples))
        if n_samples - n_validation < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) to hold out a validation set "
                f"with validation_fraction={self.validation_fraction}"
            )
        order = self._rng.permutation(n_samples)
        val_idx, train_idx = order[:n_validation], order[n_validation:]
        return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

    def _train(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> None:
        """
        Train the network on already-encoded targets Y of shape (n, k).

        Each epoch: shuffle the samples, split them into mini-batches and,
        for every mini-batch, run forward propagation, compute the loss,
        backpropagate and update the weights. The epoch loss stored in
        loss_curve_ is the sample-weighted mean of the mini-batch losses.

        Stopping: after max_iter epochs, or earlier once the monitored loss
        (validation loss if early_stopping, else training loss) has not
        improved by at least tol for n_iter_no_change consecutive epochs.
        A loss that becomes infinite or NaN raises ValueError, and a run whose
        training loss ends higher than it started raises a ConvergenceWarning.
        """
        self._validate_hyperparameters()
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = Y.shape[1]
        self._initialize_weights()
        self._initialize_optimizer()

        if self.early_stopping:
            X, Y, X_val, Y_val = self._train_validation_split(X, Y)
        n_samples = X.shape[0]
        batch_size = self._resolve_batch_size(n_samples)

        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        best_loss = np.inf
        best_parameters = None
        epochs_without_improvement = 0
        stopped_early = False

        for _ in range(self.max_iter):
            order = self._rng.permutation(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                batch = order[start:start + batch_size]
                X_batch, Y_batch = X[batch], Y[batch]

                z1, a1, z2, y_pred = self._forward_propagation(X_batch)
                epoch_loss += self._compute_loss(Y_batch, y_pred) * len(batch)
                gradients = self._backward_propagation(
                    X_batch, Y_batch, z1, a1, z2, y_pred
                )
                self._update_parameters(gradients)

            self.loss_curve_.append(epoch_loss / n_samples)

            if self.early_stopping:
                val_pred = self._forward_propagation(X_val)[3]
                self.validation_loss_curve_.append(self._compute_loss(Y_val, val_pred))
                monitored_loss = self.validation_loss_curve_[-1]
            else:
                monitored_loss = self.loss_curve_[-1]

            # An infinite or NaN loss means the weights have overflowed: stop
            # with a clear message instead of returning a model that predicts NaN.
            if not (np.isfinite(self.loss_curve_[-1]) and np.isfinite(monitored_loss)):
                raise ValueError(
                    f"{type(self).__name__}: the loss became infinite or NaN at "
                    f"epoch {len(self.loss_curve_)}, so training diverged. Use a "
                    "smaller learning_rate or standardise the features."
                )

            if monitored_loss < best_loss - self.tol:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                if self.early_stopping:
                    best_parameters = tuple(p.copy() for p in self._parameters())

            if epochs_without_improvement >= self.n_iter_no_change:
                stopped_early = True
                break

        self.n_iter_ = len(self.loss_curve_)
        if self.early_stopping:
            # Keep the weights of the epoch with the lowest validation loss.
            self.W1_, self.b1_, self.W2_, self.b2_ = best_parameters
        if self.loss_curve_[-1] > self.loss_curve_[0]:
            # The "no improvement" rule also ends a run whose loss is exploding,
            # which would otherwise look like a normal, converged fit.
            warnings.warn(
                f"{type(self).__name__}: the training loss went up from "
                f"{self.loss_curve_[0]:.3g} to {self.loss_curve_[-1]:.3g}, so "
                "training diverged. Use a smaller learning_rate.",
                ConvergenceWarning,
            )
        elif not stopped_early:
            warnings.warn(
                f"{type(self).__name__}: reached max_iter={self.max_iter} and the "
                "loss is still improving. Consider increasing max_iter.",
                ConvergenceWarning,
            )

    # ------------------------------------------------------------------
    # Methods each estimator must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform backpropagation to compute gradients.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
