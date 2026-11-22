"""
Additional tests - Gradients, optimizers, early stopping, validation, API.

The most important one is the gradient check: it compares the gradients
computed by backpropagation with a numerical estimate obtained by finite
differences, for every loss / output layer / activation combination.

Run with: pytest tests/test_extra.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from activations import identity, identity_derivative, logistic, relu_derivative, softmax
from slp_base import ConvergenceWarning, NotFittedError
from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor

ACTIVATIONS = ["identity", "logistic", "tanh", "relu"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def initialize(model, n_features, n_outputs, seed=0):
    """Give a model random weights (and non-zero biases) without training it."""
    model._rng = np.random.default_rng(seed)
    model.n_features_in_ = n_features
    model.n_outputs_ = n_outputs
    model._initialize_weights()
    model.b1_ = model._rng.normal(0, 0.5, model.b1_.shape)
    model.b2_ = model._rng.normal(0, 0.5, model.b2_.shape)
    return model


def loss_at_current_weights(model, X, Y):
    return model._compute_loss(Y, model._forward_propagation(X)[3])


def numerical_gradients(model, X, Y, eps=1e-6):
    """Central finite differences: dL/dw ~ (L(w + eps) - L(w - eps)) / (2 eps)."""
    gradients = []
    for param in model._parameters():
        grad = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            loss_plus = loss_at_current_weights(model, X, Y)
            param[idx] = original - eps
            loss_minus = loss_at_current_weights(model, X, Y)
            param[idx] = original
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        gradients.append(grad)
    return gradients


def assert_gradients_match(model, X, Y):
    analytic = model._backward_propagation(X, Y, *model._forward_propagation(X))
    numeric = numerical_gradients(model, X, Y)
    for name, a, n in zip(["dW1", "db1", "dW2", "db2"], analytic, numeric):
        assert a.shape == n.shape, name
        relative_error = np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n))
        assert relative_error < 1e-6, f"{name}: relative error {relative_error:.2e}"


@pytest.fixture
def breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


# ---------------------------------------------------------------------------
# Gradient check (backpropagation vs finite differences)
# ---------------------------------------------------------------------------


class TestGradientCheck:

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_binary_classifier(self, activation):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(8, 3))
        Y = rng.integers(0, 2, size=(8, 1)).astype(float)
        model = initialize(SimpleSLPClassifier(hidden_layer_size=4, activation=activation), 3, 1)
        assert_gradients_match(model, X, Y)

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_multiclass_classifier(self, activation):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(8, 3))
        Y = np.eye(3)[rng.integers(0, 3, size=8)]
        model = initialize(SimpleSLPClassifier(hidden_layer_size=4, activation=activation), 3, 3)
        assert_gradients_match(model, X, Y)

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    @pytest.mark.parametrize("n_outputs", [1, 2])
    def test_regressor(self, activation, n_outputs):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(8, 3))
        Y = rng.normal(size=(8, n_outputs))
        model = initialize(SimpleSLPRegressor(hidden_layer_size=4, activation=activation), 3, n_outputs)
        assert_gradients_match(model, X, Y)


# ---------------------------------------------------------------------------
# Activations: extra numerical properties
# ---------------------------------------------------------------------------


class TestActivationExtras:

    def test_identity(self):
        z = np.array([-2.0, 0.0, 3.5])
        np.testing.assert_array_equal(identity(z), z)
        np.testing.assert_array_equal(identity_derivative(z), [1.0, 1.0, 1.0])

    def test_logistic_no_overflow(self):
        with np.errstate(over="raise"):
            result = logistic(np.array([-1000.0, 1000.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])

    def test_relu_derivative_at_zero_is_zero(self):
        assert relu_derivative(np.array([0.0]))[0] == 0.0

    def test_softmax_shift_invariance(self):
        z = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(softmax(z), softmax(z + 100.0))


# ---------------------------------------------------------------------------
# Optimizers and mini-batches
# ---------------------------------------------------------------------------


class TestOptimizers:

    def test_sgd_step_is_minus_learning_rate_times_gradient(self):
        """One full-batch SGD step without momentum: w_new = w - lr * dL/dw."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 3.0])
        lr = 0.1

        model = SimpleSLPRegressor(
            hidden_layer_size=5, solver="sgd", momentum=0.0, batch_size=30,
            learning_rate=lr, max_iter=1, random_state=0,
        )
        with pytest.warns(ConvergenceWarning):
            model.fit(X, y)

        # Same initial weights (same seed), gradient computed by hand
        reference = SimpleSLPRegressor(hidden_layer_size=5)
        reference._rng = np.random.default_rng(0)
        reference.n_features_in_, reference.n_outputs_ = 4, 1
        reference._initialize_weights()
        Y_scaled = ((y - y.mean()) / y.std()).reshape(-1, 1)
        gradients = reference._backward_propagation(
            X, Y_scaled, *reference._forward_propagation(X)
        )
        for after, before, grad in zip(model._parameters(), reference._parameters(), gradients):
            np.testing.assert_allclose(after, before - lr * grad, atol=1e-12)

    def test_first_adam_step_moves_each_weight_by_learning_rate(self):
        """After bias correction, Adam's first step is lr * sign(gradient)."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        Y = rng.normal(size=(30, 1))
        model = initialize(SimpleSLPRegressor(hidden_layer_size=5, learning_rate=0.01), 4, 1)
        model._initialize_optimizer()
        before = [p.copy() for p in model._parameters()]
        gradients = model._backward_propagation(X, Y, *model._forward_propagation(X))
        model._update_parameters(gradients)
        for after, old, grad in zip(model._parameters(), before, gradients):
            np.testing.assert_allclose(after - old, -0.01 * np.sign(grad), atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 16, 50, 1000])
    def test_number_of_updates_per_epoch(self, batch_size):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(50, 3)), rng.integers(0, 2, 50)
        clf = SimpleSLPClassifier(
            hidden_layer_size=5, batch_size=batch_size, max_iter=3, random_state=0
        )
        with pytest.warns(ConvergenceWarning):
            clf.fit(X, y)
        updates_per_epoch = int(np.ceil(50 / min(batch_size, 50)))
        assert clf._adam_t == clf.n_iter_ * updates_per_epoch

    @pytest.mark.parametrize(
        "solver, momentum, learning_rate",
        [("sgd", 0.0, 0.1), ("sgd", 0.9, 0.01), ("adam", 0.9, 0.001)],
    )
    def test_every_solver_learns(self, breast_cancer, solver, momentum, learning_rate):
        X_train, X_test, y_train, y_test = breast_cancer
        clf = SimpleSLPClassifier(
            hidden_layer_size=20, solver=solver, momentum=momentum,
            learning_rate=learning_rate, batch_size=32, max_iter=50, random_state=0,
        )
        clf.fit(X_train, y_train)
        assert clf.loss_curve_[-1] < 0.5 * clf.loss_curve_[0]
        assert clf.score(X_test, y_test) > 0.93


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------


class TestStopping:

    def test_warns_when_max_iter_reached(self):
        X, y = np.random.default_rng(0).normal(size=(20, 2)), np.arange(20) % 2
        with pytest.warns(ConvergenceWarning):
            SimpleSLPClassifier(hidden_layer_size=3, max_iter=2, random_state=0).fit(X, y)

    def test_stops_when_training_loss_plateaus(self):
        """With a huge tol no epoch counts as an improvement after the first."""
        X, y = np.random.default_rng(0).normal(size=(20, 2)), np.arange(20) % 2
        clf = SimpleSLPClassifier(
            hidden_layer_size=3, max_iter=100, tol=10.0, n_iter_no_change=5, random_state=0
        )
        clf.fit(X, y)
        assert clf.n_iter_ == 1 + 5
        assert len(clf.loss_curve_) == clf.n_iter_
        assert clf.validation_loss_curve_ == []

    @pytest.mark.parametrize("early_stopping", [False, True])
    def test_non_finite_loss_raises(self, early_stopping):
        """Inputs so large that the loss overflows give a clear error, not NaN weights."""
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(50, 3)) * 1e300, rng.normal(size=50)
        reg = SimpleSLPRegressor(
            activation="relu", max_iter=20, early_stopping=early_stopping, random_state=0
        )
        with np.errstate(all="ignore"), pytest.raises(ValueError, match="diverged"):
            reg.fit(X, y)

    def test_warns_when_training_loss_goes_up(self):
        """A learning rate far too high makes the loss explode; the user is told."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = X @ np.array([1.0, 2.0, 3.0])
        reg = SimpleSLPRegressor(solver="sgd", learning_rate=50.0, max_iter=50, random_state=0)
        with pytest.warns(ConvergenceWarning, match="went up"):
            reg.fit(X, y)

    def test_early_stopping_uses_validation_loss(self, breast_cancer):
        X_train, X_test, y_train, y_test = breast_cancer
        clf = SimpleSLPClassifier(
            hidden_layer_size=50, learning_rate=0.01, max_iter=1000,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=10,
            random_state=0,
        )
        clf.fit(X_train, y_train)
        assert clf.n_iter_ < 1000
        assert len(clf.validation_loss_curve_) == clf.n_iter_
        # It stopped because none of the last n_iter_no_change epochs beat the
        # best earlier validation loss by at least tol
        curve = clf.validation_loss_curve_
        assert min(curve[-10:]) >= min(curve[:-10]) - clf.tol
        assert clf.score(X_test, y_test) > 0.93

    def test_early_stopping_restores_best_weights(self):
        """Final weights must be the ones with the lowest validation loss."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = (X[:, 0] + rng.normal(scale=2.0, size=60) > 0).astype(int)  # noisy -> overfits
        params = dict(
            hidden_layer_size=100, learning_rate=0.05, max_iter=500,
            early_stopping=True, n_iter_no_change=20, random_state=0,
            validation_fraction=0.3,
        )
        clf = SimpleSLPClassifier(**params).fit(X, y)

        # Rebuild the same validation split (same seed, same order of rng calls)
        twin = SimpleSLPClassifier(**params)
        twin._rng = np.random.default_rng(0)
        twin.n_features_in_, twin.n_outputs_ = 5, 1
        twin._initialize_weights()
        Y = y.reshape(-1, 1).astype(float)
        _, _, X_val, Y_val = twin._train_validation_split(X, Y)

        final_val_loss = clf._compute_loss(Y_val, clf._forward_propagation(X_val)[3])
        assert final_val_loss == pytest.approx(min(clf.validation_loss_curve_))
        assert final_val_loss < clf.validation_loss_curve_[-1]  # it did overfit


# ---------------------------------------------------------------------------
# Parameter and input validation
# ---------------------------------------------------------------------------


class TestValidation:

    @pytest.mark.parametrize(
        "params",
        [
            {"hidden_layer_size": 0},
            {"hidden_layer_size": 2.5},
            {"activation": "sigmoid"},
            {"learning_rate": 0},
            {"learning_rate": -0.1},
            {"max_iter": 0},
            {"solver": "lbfgs"},
            {"momentum": 1.0},
            {"batch_size": 0},
            {"batch_size": "full"},
            {"validation_fraction": 1.0},
            {"n_iter_no_change": 0},
            {"tol": -1.0},
            {"early_stopping": "yes"},
            {"random_state": -1},
        ],
    )
    def test_invalid_hyperparameters_raise(self, params):
        X, y = np.random.default_rng(0).normal(size=(10, 2)), np.arange(10) % 2
        with pytest.raises(ValueError):
            SimpleSLPClassifier(**params).fit(X, y)
        with pytest.raises(ValueError):
            SimpleSLPRegressor(**params).fit(X, y.astype(float))

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            SimpleSLPClassifier().predict(np.zeros((2, 2)))
        with pytest.raises(NotFittedError):
            SimpleSLPRegressor().predict(np.zeros((2, 2)))

    def test_bad_inputs_raise(self):
        clf = SimpleSLPClassifier(hidden_layer_size=3, max_iter=5, random_state=0)
        with pytest.raises(ValueError):
            clf.fit(np.zeros(10), np.zeros(10))  # 1D X
        with pytest.raises(ValueError):
            clf.fit(np.zeros((10, 2)), np.zeros(9))  # length mismatch
        with pytest.raises(ValueError):
            clf.fit(np.full((10, 2), np.nan), np.zeros(10))  # NaN

    def test_wrong_number_of_features_at_predict(self):
        X, y = np.random.default_rng(0).normal(size=(20, 3)), np.arange(20) % 2
        clf = SimpleSLPClassifier(hidden_layer_size=3, max_iter=5, random_state=0)
        with pytest.warns(ConvergenceWarning):
            clf.fit(X, y)
        with pytest.raises(ValueError):
            clf.predict(np.zeros((2, 4)))


# ---------------------------------------------------------------------------
# Interface and behaviour
# ---------------------------------------------------------------------------


class TestInterface:

    def test_get_and_set_params(self):
        clf = SimpleSLPClassifier(hidden_layer_size=7, solver="sgd")
        params = clf.get_params()
        assert params["hidden_layer_size"] == 7 and params["solver"] == "sgd"
        assert set(params) >= {"activation", "learning_rate", "max_iter", "random_state"}
        assert clf.set_params(learning_rate=0.5) is clf
        assert clf.learning_rate == 0.5
        with pytest.raises(ValueError):
            clf.set_params(not_a_parameter=1)

    def test_fit_returns_self(self):
        X, y = np.random.default_rng(0).normal(size=(20, 2)), np.arange(20) % 2
        clf = SimpleSLPClassifier(hidden_layer_size=3, max_iter=50, random_state=0)
        reg = SimpleSLPRegressor(hidden_layer_size=3, max_iter=50, random_state=0)
        assert clf.fit(X, y) is clf
        assert reg.fit(X, y.astype(float)) is reg

    def test_string_labels(self):
        X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
        y = np.array(["cat", "cat", "cat", "dog", "dog", "dog"])
        clf = SimpleSLPClassifier(
            hidden_layer_size=5, learning_rate=0.05, max_iter=200, random_state=0
        ).fit(X, y)
        assert list(clf.classes_) == ["cat", "dog"]
        assert list(clf.predict(X)) == list(y)

    def test_binary_uses_one_sigmoid_output(self, breast_cancer):
        X_train, _, y_train, _ = breast_cancer
        clf = SimpleSLPClassifier(hidden_layer_size=5, max_iter=50, random_state=0).fit(X_train, y_train)
        assert clf.n_outputs_ == 1 and clf.W2_.shape == (5, 1)

    def test_single_class_probabilities(self):
        X = np.random.default_rng(0).normal(size=(10, 2))
        clf = SimpleSLPClassifier(hidden_layer_size=3, max_iter=20, random_state=0)
        clf.fit(X, np.full(10, 7))
        np.testing.assert_array_equal(clf.predict_proba(X), np.ones((10, 1)))
        assert np.all(clf.predict(X) == 7)

    def test_iris_multiclass(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        scaler = StandardScaler()
        clf = SimpleSLPClassifier(
            hidden_layer_size=20, activation="tanh", learning_rate=0.01,
            max_iter=300, random_state=0,
        ).fit(scaler.fit_transform(X_train), y_train)
        proba = clf.predict_proba(scaler.transform(X_test))
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
        assert clf.score(scaler.transform(X_test), y_test) > 0.9

    def test_breast_cancer_binary(self, breast_cancer):
        X_train, X_test, y_train, y_test = breast_cancer
        clf = SimpleSLPClassifier(hidden_layer_size=30, learning_rate=0.01, random_state=0)
        clf.fit(X_train, y_train)
        assert clf.score(X_test, y_test) > 0.95

    def test_regressor_predicts_in_original_units(self):
        """Target scaling is internal: predictions come back in y's units."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        y = 1000.0 * (X[:, 0] - 0.5 * X[:, 1]) + 5000.0
        reg = SimpleSLPRegressor(
            hidden_layer_size=10, learning_rate=0.01, max_iter=300, random_state=0
        ).fit(X, y)
        assert reg.score(X, y) > 0.98
        assert abs(reg.predict(X).mean() - y.mean()) < 0.05 * y.std()

    def test_multi_output_regression(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        Y = np.column_stack([X[:, 0] + X[:, 1], 10.0 * X[:, 2] - 3.0])
        reg = SimpleSLPRegressor(
            hidden_layer_size=20, learning_rate=0.01, max_iter=300, random_state=0
        ).fit(X, Y)
        assert reg.predict(X).shape == (200, 2)
        assert reg.n_outputs_ == 2
        assert reg.score(X, Y) > 0.95

    def test_scores_match_sklearn_metrics(self):
        """Our score() must equal sklearn's accuracy_score / r2_score."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y_class = (X[:, 0] > 0).astype(int)
        y_reg = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=100)
        Y_reg = np.column_stack([y_reg, 5 * X[:, 1] ** 2])

        clf = SimpleSLPClassifier(hidden_layer_size=5, max_iter=30, random_state=0)
        reg = SimpleSLPRegressor(hidden_layer_size=5, max_iter=30, random_state=0)
        reg2 = SimpleSLPRegressor(hidden_layer_size=5, max_iter=30, random_state=0)
        with pytest.warns(ConvergenceWarning):
            clf.fit(X, y_class)
        with pytest.warns(ConvergenceWarning):
            reg.fit(X, y_reg)
        with pytest.warns(ConvergenceWarning):
            reg2.fit(X, Y_reg)

        assert clf.score(X, y_class) == pytest.approx(accuracy_score(y_class, clf.predict(X)))
        assert reg.score(X, y_reg) == pytest.approx(r2_score(y_reg, reg.predict(X)))
        assert reg2.score(X, Y_reg) == pytest.approx(r2_score(Y_reg, reg2.predict(X)))
