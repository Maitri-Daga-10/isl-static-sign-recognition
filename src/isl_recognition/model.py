"""The Phase 1 baseline classifier."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_classifier(seed: int = 42):
    """Return a scaled linear SVM suitable for compact landmark vectors."""

    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=20000),
    )
