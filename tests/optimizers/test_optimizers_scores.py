import pytest
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from siapy.optimizers.scorers import Scorer


def test_init_cross_validator_scorer(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    scorer = Scorer.init_cross_validator_scorer()
    score = scorer(DecisionTreeClassifier(random_state=0), X, y)
    assert score == pytest.approx(0.81)


def test_init_cross_validator_scorer_repeated_kfold(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    scorer1 = Scorer.init_cross_validator_scorer(
        scoring="f1", cv=RepeatedKFold(n_splits=3, n_repeats=5, random_state=0)
    )
    scorer2 = Scorer.init_cross_validator_scorer(scoring="f1", cv="RepeatedKFold")
    score1 = scorer1(DecisionTreeClassifier(random_state=0), X, y)
    score2 = scorer2(DecisionTreeClassifier(random_state=0), X, y)
    assert score1 == score2


def test_init_hold_out_scorer(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    scorer = Scorer.init_hold_out_scorer(test_size=0.3)
    score = scorer(DecisionTreeClassifier(random_state=0), X, y)
    assert score == pytest.approx(0.87, rel=1e-2)


def test_hold_out_validation_with_manual_split(mock_sklearn_dataset):
    X, y = mock_sklearn_dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    scorer = Scorer.init_hold_out_scorer(scoring="f1")
    score = scorer(
        DecisionTreeClassifier(random_state=0), X_train, y_train, X_val, y_val
    )
    assert score == pytest.approx(0.87, rel=1e-2)
