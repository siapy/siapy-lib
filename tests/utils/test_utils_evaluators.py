import logging

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

from siapy.utils.evaluators import (
    cross_validation,
    hold_out_validation,
)


def test_cross_validation(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    mean_score = cross_validation(model=SVC(random_state=0), X=X, y=y)
    assert mean_score == pytest.approx(0.9)
    mean_score = cross_validation(model=DummyClassifier(), X=X, y=y)
    assert round(mean_score, 2) == pytest.approx(0.52)


def test_cross_validation_with_kfold(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    mean_score = cross_validation(model=SVC(random_state=0), X=X, y=y, cv=kf)
    assert mean_score == pytest.approx(0.92)


def test_cross_validation_with_custom_scorer(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    custom_scorer = make_scorer(accuracy_score)
    mean_score = cross_validation(
        model=SVC(random_state=0), X=X, y=y, scoring=custom_scorer
    )
    assert mean_score == pytest.approx(0.9)


def test_cross_validation_with_x_val_y_val(mock_sklearn_dataset, caplog):
    X, y = mock_sklearn_dataset
    caplog.set_level(logging.INFO)  # Set the logging level to INFO
    mean_score = cross_validation(model=SVC(random_state=0), X=X, y=y, X_val=X)
    assert mean_score == pytest.approx(0.9)
    assert (
        "Specification of X_val and y_val is redundant for cross_validation."
        "These parameters are ignored." in caplog.text
    )


def test_hold_out_validation(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    score = hold_out_validation(model=SVC(random_state=0), X=X, y=y, random_state=0)
    assert score == pytest.approx(0.95)
    score = hold_out_validation(model=DummyClassifier(), X=X, y=y, random_state=0)
    assert score == pytest.approx(0.40)


def test_hold_out_validation_with_manual_validation_set(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    score = hold_out_validation(
        model=SVC(random_state=0), X=X_train, y=y_train, X_val=X_val, y_val=y_val
    )
    assert score == pytest.approx(0.95)


def test_hold_out_validation_with_incomplete_manual_validation_set(
    mock_sklearn_dataset,
):
    X, y = mock_sklearn_dataset
    X_train, X_val, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    with pytest.raises(
        ValueError,
        match="To manually define validation set, both X_val and y_val must be specified.",
    ):
        hold_out_validation(
            model=SVC(random_state=0), X=X_train, y=y_train, X_val=X_val
        )


def test_hold_out_validation_with_custom_scorer_func(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    custom_scorer = make_scorer(accuracy_score)
    score = hold_out_validation(
        model=SVC(random_state=0), X=X, y=y, scoring=custom_scorer, random_state=0
    )
    assert score == pytest.approx(0.95)


def test_hold_out_validation_with_custom_scorer_str(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    score = hold_out_validation(
        model=SVC(random_state=0), X=X, y=y, scoring="accuracy", random_state=0
    )
    assert score == pytest.approx(0.95)


def test_hold_out_validation_with_stratify(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    score = hold_out_validation(
        model=SVC(random_state=0), X=X, y=y, stratify=y, random_state=0
    )
    assert score == pytest.approx(0.95)
