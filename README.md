# One-hidden-layer neural network from scratch

This is a NumPy implementation of a neural network with one hidden layer, for both classification (`SimpleSLPClassifier`) and regression (`SimpleSLPRegressor`). Both models follow the interface of scikit-learn's `MLPClassifier` and `MLPRegressor`.

It is the course project for *Mathematics for Machine Learning*. The report, with the maths, the demonstrations and the comparison with scikit-learn, is **[`report.ipynb`](report.ipynb)**.

## Quick start

```bash
pip install -r requirements.txt
pytest                      # 118 tests: the 54 provided + 64 in tests/test_extra.py
jupyter notebook report.ipynb
```

```python
from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor

clf = SimpleSLPClassifier(hidden_layer_size=30, activation="relu", learning_rate=0.01,
                          early_stopping=True, random_state=0)
clf.fit(X_train, y_train)          # returns self
clf.predict_proba(X_test)          # (n_samples, n_classes)
clf.score(X_test, y_test)          # accuracy
clf.loss_curve_                    # training loss per epoch

reg = SimpleSLPRegressor(hidden_layer_size=20, solver="sgd", momentum=0.9, batch_size=32)
reg.fit(X_train, y_train).score(X_test, y_test)   # R²
```

Tested with Python 3.12, NumPy 2.5 and scikit-learn 1.9. Running the whole notebook takes about two minutes. The California Housing dataset is downloaded once by scikit-learn.

## Requirements covered

| Requirement | Where |
|---|---|
| **Basic** | |
| Forward propagation with one hidden layer | `_forward_propagation` in `slp_classifier.py` and `slp_regressor.py` |
| Sigmoid/logistic activation (also tanh, ReLU, identity) | `activations.py` |
| Output layers: sigmoid (binary), linear (regression) | `slp_classifier.py`, `slp_regressor.py` |
| Backpropagation | `_backward_propagation` (output error) and `BaseSLPEstimator._gradients_from_output_delta` (hidden layer) |
| Stochastic gradient descent | `solver="sgd"`, `BaseSLPEstimator._update_parameters` |
| scikit-learn conventions | `fit` returns `self`; `predict`, `predict_proba`, `score`; learned attributes end in `_`; `get_params` / `set_params`; hyper-parameters validated in `fit` |
| Loss tracking | `loss_curve_` (and `validation_loss_curve_` with early stopping) |
| Tests pass | `pytest`: all 118 pass |
| **Advanced** | |
| Mini-batch / full-batch gradient descent | `batch_size` (`"auto"` = min(200, n); `1` = pure SGD; `>= n` = full batch) |
| Vectorised NumPy (no loops over samples or weights) | the only loops are over epochs, mini-batches and the 4 parameter arrays |
| Early stopping | `early_stopping`, `validation_fraction`, `n_iter_no_change`, `tol`; the best weights are restored |
| Code quality | docstrings with the maths, type hints, `ValueError` on bad hyper-parameters or inputs, `NotFittedError`, `ConvergenceWarning`, a clear error or warning when training diverges |
| **Report** | `report.ipynb`: diabetes, breast cancer, California Housing and two moons; comparison with sklearn; loss curves; accuracy / R² |
| **Bonus** | |
| Multi-class with softmax output | automatic when there are more than 2 classes; demonstrated on `load_digits` and `load_iris` |
| Momentum and adaptive gradient descent | `solver="sgd"` with `momentum`; `solver="adam"` (default) |
| Additional tests | `tests/test_extra.py`: gradient check against finite differences, optimizer update rules, early stopping and weight restoring, divergence handling, validation, interface |
| Arbitrary depth | *not implemented* |

## Hyper-parameters

| Parameter | Default | Meaning |
|---|---|---|
| `hidden_layer_size` | 100 | number of hidden neurons |
| `activation` | `"logistic"` | hidden activation: `"identity"`, `"logistic"`, `"tanh"`, `"relu"` |
| `solver` | `"adam"` | `"sgd"` (with momentum) or `"adam"` |
| `learning_rate` | 0.001 | step size |
| `momentum` | 0.9 | momentum for `"sgd"` (0 gives plain SGD) |
| `batch_size` | `"auto"` | mini-batch size; `"auto"` = min(200, n_samples) |
| `max_iter` | 200 | maximum number of epochs |
| `early_stopping` | `False` | stop on the loss of a held-out validation set and restore the best weights |
| `validation_fraction` | 0.1 | share of the training data held out for early stopping |
| `n_iter_no_change` | 10 | epochs without an improvement of at least `tol` before stopping |
| `tol` | 1e-4 | minimum loss decrease that counts as an improvement |
| `random_state` | `None` | seed for the initialisation, the shuffling and the validation split |

Fitted attributes: `W1_`, `b1_`, `W2_`, `b2_`, `loss_curve_`, `validation_loss_curve_`, `n_iter_`, `n_features_in_` and `n_outputs_`. The classifier also has `classes_`, and the regressor has `y_mean_` and `y_scale_`.

## Design decisions

Each decision is justified with an experiment in Section 8 of the report.

1. **Adam is the default solver.** The provided tests use `learning_rate=0.001` with 100–300 epochs. At that rate plain SGD barely learns and fails `test_score`. Adam is also scikit-learn's default. SGD with momentum is available with `solver="sgd"`.
2. **Glorot/Xavier uniform initialisation**, the same as scikit-learn, instead of $\mathcal N(0, 0.01^2)$. Tiny weights make the hidden units almost identical, and the gradient that reaches $W_1$ very small, so learning starts slowly.
3. **The regressor standardises the targets internally** and maps its predictions back to the original units, following the project tips. Without this step it needs about 950 epochs instead of 40 on diabetes, and it fails the provided `test_score`.
4. **Binary classification uses a single sigmoid output** (`n_outputs_ = 1`), as scikit-learn does. With more than 2 classes the output becomes a softmax over k units. `predict_proba` always returns one column per class.

## Results (test set, mean ± std over 5 seeds)

| Dataset | Task | Metric | Linear baseline | Ours | scikit-learn |
|---|---|---|---|---|---|
| Diabetes | regression | R² | 0.332 | 0.337 ± 0.022 | 0.329 ± 0.027 |
| Breast cancer | binary | accuracy | 0.982 | 0.982 ± 0.000 | 0.982 ± 0.000 |
| Digits | 10 classes | accuracy | 0.967 | 0.976 ± 0.001 | 0.977 ± 0.004 |
| Iris | 3 classes | accuracy | 0.967 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| California Housing | regression | R² | 0.594 | 0.756 (20 seeds) | 0.760 (20 seeds) |
| Two moons | binary | accuracy | 0.827 | 0.927 ± 0.007 | 0.923 ± 0.006 |

The scikit-learn models use exactly the same settings (`alpha=0`, the same stopping rule). For California Housing, 5 seeds gave 0.742 ± 0.024 vs 0.774 ± 0.006. That gap is seed noise caused by a few extreme test districts, and it disappears over 20 seeds (Section 7.1 of the report).

## Layout

```
activations.py        activation functions and their derivatives
slp_base.py           shared code: validation, initialisation, training loop, optimizers, early stopping
slp_classifier.py     SimpleSLPClassifier
slp_regressor.py      SimpleSLPRegressor
report.ipynb          the report (executed, with all outputs)
tests/                test_activations.py, test_models.py, test_smoke.py (provided), test_extra.py (added)
pytest.ini            puts the project root on the import path for the tests
```
