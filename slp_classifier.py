"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from activations import logistic, softmax
from slp_base import BaseSLPEstimator

# Probabilities are clipped to [eps, 1 - eps] before taking log() so the
# cross-entropy never evaluates log(0) = -inf.
_PROBA_EPS = np.finfo(float).eps


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.

    Output layer:
    - 2 classes: one sigmoid unit giving P(y = classes_[1] | x), trained with
      binary cross-entropy (n_outputs_ = 1, as in scikit-learn).
    - k > 2 classes: k softmax units, trained with categorical cross-entropy
      on one-hot encoded targets (n_outputs_ = k).

    Class labels can be any sortable values (ints, strings, ...); they are
    stored, sorted, in classes_.
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
        Initialize the SLP classifier.

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

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None  # Unique class labels
        self.n_outputs_: Optional[int] = None  # Number of output neurons

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
        # Binary: sigmoid of the single output. Multi-class: softmax over k outputs.
        y_pred = logistic(z2) if self.n_outputs_ == 1 else softmax(z2)

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
            One-hot encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        # Output layer error. For sigmoid + binary cross-entropy and for
        # softmax + categorical cross-entropy the derivative of the mean loss
        # with respect to z2 simplifies to the same expression:
        #     dL/dz2 = (y_pred - y) / n
        # (the sigmoid'/softmax Jacobian cancels against d log(p)/dp).
        n_samples = X.shape[0]
        delta2 = (y_pred - y) / n_samples

        # Hidden layer error and weight/bias gradients (shared code)
        return self._gradients_from_output_delta(X, z1, a1, delta2)

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Binary (one output column):   L = -mean( y log p + (1 - y) log(1 - p) )
        Multi-class (one-hot, k cols): L = -mean over samples of sum_c y_c log p_c

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # Clip predictions to avoid log(0)
        p = np.clip(y_pred, _PROBA_EPS, 1.0 - _PROBA_EPS)
        n_samples = y_true.shape[0]
        if y_true.shape[1] == 1:
            log_likelihood = y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
        else:
            log_likelihood = y_true * np.log(p)
        return float(-np.sum(log_likelihood) / n_samples)

    def _encode_targets(self, y_indices: NDArray[np.int_]) -> NDArray[np.floating]:
        """
        Turn class indices (0..k-1) into the network's target matrix.

        2 classes -> column of 0/1 (1 means classes_[1]), shape (n, 1)
        k classes -> one-hot matrix, shape (n, k)
        """
        n_classes = len(self.classes_)
        if n_classes <= 2:
            # With a single class this column is all zeros; predict_proba
            # handles that case separately.
            return (y_indices == 1).astype(float).reshape(-1, 1)
        return np.eye(n_classes)[y_indices]

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array of labels, got shape {y.shape}")
        self._check_same_length(X, y)

        # Identify unique classes and encode y (sigmoid for 2 classes, softmax for more)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        Y = self._encode_targets(y_indices)

        # Seed, weight initialization and the training loop live in the base class
        self._train(X, Y)
        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        X = self._validate_X_for_prediction(X)
        output = self._forward_propagation(X)[3]

        if len(self.classes_) == 1:
            # Only one class was seen during fit: it has probability 1.
            return np.ones((X.shape[0], 1))
        if self.n_outputs_ == 1:
            # Sigmoid gives P(classes_[1]); P(classes_[0]) is the complement.
            return np.hstack([1.0 - output, output])
        return output

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        # Get probabilities and return class with highest probability
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        self._check_same_length(y_pred, y)
        return float(np.mean(y_pred == y))
