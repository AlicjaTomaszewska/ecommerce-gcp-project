import numpy as np

class HurdleModel:
    def __init__(self, classifier, classifier_scaler, regressor, regressor_scaler, feature_names=None):
        self.classifier = classifier
        self.classifier_scaler = classifier_scaler
        self.regressor = regressor
        self.regressor_scaler = regressor_scaler
        self.feature_names = feature_names

    def _apply(self, scaler, X):
        return scaler.transform(X) if scaler is not None else X

    def predict_proba_nonzero(self, X):
        Xc = self._apply(self.classifier_scaler, X)
        return self.classifier.predict_proba(Xc)[:, 1]

    def predict_positive_count(self, X):
        Xr = self._apply(self.regressor_scaler, X)
        return np.clip(self.regressor.predict(Xr), 0, None)

    def predict(self, X, threshold=None):
        p = self.predict_proba_nonzero(X)
        cnt = self.predict_positive_count(X)
        if threshold is not None:
            return np.where(p >= threshold, cnt, 0.0)
        return p * cnt
