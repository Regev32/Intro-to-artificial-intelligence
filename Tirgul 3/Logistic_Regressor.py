import numpy as np
from typing import Optional


class LogisticRegressor:
    """
    Simple binary logistic regression trained with gradient descent.

    API is loosely similar to sklearn:
    - fit(X, y, X_val=None, y_val=None)
    - predict_proba(X)
    - predict(X)
    - decision_function(X)

    After fitting you have:
    - weights_ : np.ndarray of shape (n_features,)
    - bias_    : float
    - loss_history_      : list of training losses per epoch
    - val_loss_history_  : list of validation losses per epoch (or None)
    """

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 2000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.weights_: Optional[np.ndarray] = None  # shape (n_features,)
        self.bias_: Optional[float] = None
        self.loss_history_: list[float] = []
        self.val_loss_history_: Optional[list[float]] = None

    # ---- internal helpers ----
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _bce_loss(y: np.ndarray, p: np.ndarray) -> float:
        """
        Binary cross-entropy loss.

        y: (n_samples,) with values 0 or 1
        p: (n_samples,) probabilities P(y=1|x)
        """
        eps = 1e-10
        return -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        z = X @ self.weights_ + self.bias_
        return self._sigmoid(z)

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Gradients of the loss w.r.t. weights and bias.
        """
        n_samples = len(y)
        error = p - y
        grad_w = X.T @ error / n_samples
        grad_b = np.sum(error) / n_samples
        return grad_w, grad_b

    # ---- public API ----
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LogisticRegressor":
        """
        Train the logistic regression model using gradient descent.

        X: (n_samples, n_features)
        y: (n_samples,) with values 0 or 1

        X_val, y_val (optional): validation set for tracking val loss.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val).ravel()
            self.val_loss_history_ = []
        else:
            self.val_loss_history_ = None

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            # forward on train
            p_train = self._predict_proba_internal(X)
            train_loss = self._bce_loss(y, p_train)
            self.loss_history_.append(train_loss)

            # optional validation loss
            if X_val is not None and y_val is not None:
                z_val = X_val @ self.weights_ + self.bias_
                p_val = self._sigmoid(z_val)
                val_loss = self._bce_loss(y_val, p_val)
                self.val_loss_history_.append(val_loss)

            # gradients and update (using training set only)
            grad_w, grad_b = self._compute_gradients(X, y, p_train)
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(y=1|x) for each sample.

        Output shape: (n_samples,)
        """
        X = np.asarray(X)
        return self._predict_proba_internal(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return w^T x + b (the logit).
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        X = np.asarray(X)
        return X @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return hard class labels (0/1) using given threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
