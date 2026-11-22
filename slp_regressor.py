"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPRegressor.

    The output layer is linear (identity activation) and the loss is the
    mean squared error. Multi-output regression (y of shape (n, k)) is
    supported.

    Target scaling: as recommended in the project tips, the targets are
    standardized inside fit() (y_scaled = (y - y_mean_) / y_scale_) and
    predict() maps the network output back to the original units. This makes
    the default learning rates work whatever the scale of y (e.g. diabetes
    targets go from 25 to 346). It does not change R², and loss_curve_ is the
    MSE on the standardized targets.
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
        Initialize the SLP regressor.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations (epochs)
        random_state : int or None
            Random seed for reproducibility
        solver, momentum, batch_size, early_stopping, validation_fraction,
        n_iter_no_change, tol :
            Optimization settings, see BaseSLPEstimator
        """
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter, random_state,
            solver=solver, momentum=momentum, batch_size=batch_size,
            early_stopping=early_stopping, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol,
        )

        # Regressor-specific attributes (set in fit)
        self.y_mean_: Optional[NDArray[np.floating]] = None  # Target mean, shape (k,)
        self.y_scale_: Optional[NDArray[np.floating]] = None  # Target std, shape (k,)

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        z1, a1, z2, y_pred : tuple of arrays
            Intermediate values for backpropagation
        """
        activation, _ = self._get_activation_function()

        z1 = X @ self.W1_ + self.b1_  # (n, h)
        a1 = activation(z1)  # (n, h)
        z2 = a1 @ self.W2_ + self.b2_  # (n, k)
        y_pred = z2  # Identity output activation for regression

        return z1, a1, z2, y_pred

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

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            Target values
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        # Output layer error for MSE with identity output (y_pred = z2):
        #     L = 1/(n k) * sum (y_pred - y)^2
        #     dL/dz2 = 2 (y_pred - y) / (n k)
        n_samples, n_outputs = y.shape
        delta2 = 2.0 * (y_pred - y) / (n_samples * n_outputs)

        # Hidden layer error and weight/bias gradients (shared code)
        return self._gradients_from_output_delta(X, z1, a1, delta2)

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute mean squared error loss.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        loss : float
            MSE loss
        """
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _validate_y(y: NDArray) -> NDArray[np.floating]:
        """Convert y to a float array of shape (n,) or (n, k) and check it is finite."""
        y = np.asarray(y, dtype=float)
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values")
        return y

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = self._validate_X(X)
        y = self._validate_y(y)
        self._check_same_length(X, y)
        Y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Standardize the targets; a constant target has std 0, so use 1 there.
        self.y_mean_ = Y.mean(axis=0)
        scale = Y.std(axis=0)
        self.y_scale_ = np.where(scale > 0, scale, 1.0)

        self._train(X, (Y - self.y_mean_) / self.y_scale_)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        X = self._validate_X_for_prediction(X)
        y_scaled = self._forward_propagation(X)[3]
        y_pred = y_scaled * self.y_scale_ + self.y_mean_  # back to original units
        # Single output -> 1D array, like scikit-learn
        return y_pred.ravel() if self.n_outputs_ == 1 else y_pred

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² score on the given test data.

        R² = 1 - SS_res / SS_tot, with SS_res = sum (y - y_pred)^2 and
        SS_tot = sum (y - mean(y))^2. For several outputs the R² of each
        output is averaged (scikit-learn's 'uniform_average').

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values

        Returns:
        --------
        score : float
            R² score
        """
        y = self._validate_y(y)
        y_pred = self.predict(X)
        self._check_same_length(y_pred, y)
        Y = y.reshape(len(y), -1)
        Y_pred = y_pred.reshape(len(y_pred), -1)

        ss_res = np.sum((Y - Y_pred) ** 2, axis=0)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
        # A constant target has ss_tot = 0: R² is 1 for a perfect fit, else 0
        # (same convention as sklearn.metrics.r2_score).
        safe_ss_tot = np.where(ss_tot > 0, ss_tot, 1.0)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / safe_ss_tot, (ss_res == 0).astype(float))
        return float(np.mean(r2))
