from typing import Dict, List

import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate the Variance Inflation Factor values
    associated with different features.

    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with VIF values.
    """

    # add constant for intercept
    df_with_const = sm.add_constant(feature_df)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_with_const.values, i + 1)
        for i in range(len(feature_df.columns))
    ]
    return vif_data


def hyperparameter_tune_rf(
    feature_training_data: pd.DataFrame,
    y_training_data: pd.Series,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
    scoring: str = "accuracy",
    param_grid: Dict[str, List[int]] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
) -> RandomForestClassifier:
    """
    This function allows us to run a Grid Search with Cross Validation to test out
    multiple hyperparameters and determine the best ones.

    Parameters
    ---------
    feature_training_data: pd.DataFrame
        Training data with relevant features.
    y_training_data: pd.Series
        The y training data.
    cv: int
        # of cross-validation folds
    n_jobs: int
        Number of jobs to run in parallel, (-1 means us all processors).
    verbose: int
        The number of messages we want displayed.
    scoring: str
        How we want to evaluate model performance.
    param_grid: dict[str:List[int]]
        Dictionary of parameter names and parameter settings to try.

    Returns
    ------
    RandomForestClassifier
    """

    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring=scoring,
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(feature_training_data, y_training_data)

    return grid_search.best_estimator_


def evaluate_random_forest(
    best_rf: RandomForestClassifier,
    input_data: pd.DataFrame,
    y_data: pd.Series,
    train_or_validation: str,
) -> None:
    """
    This function allows us to quickly evaluate model performance for different input data.

    Parameters
    ----------
    best_rf: RandomForestClassifier
       Output from hyperparameter_tune_rf.
    input_data: pd.DataFrame
       Input data we want to generate predictions on.
    y_data: pd.Series
        Y data we want to compare our predictions to.
    train_or_validation: str
        Specificies if this is a training or validation dataset.

    Returns
    ------
    None

    """

    features = input_data.columns.tolist()

    # generate predictions
    y_pred = best_rf.predict(input_data)

    # Evaluate the model
    accuracy = accuracy_score(y_data, y_pred)
    print(
        f"Accuracy on {train_or_validation} data with features {features}: {accuracy:.4f}"
    )

    print(f"{train_or_validation.title()} Classification Report:\n" + "-" * 30)
    print(classification_report(y_data, y_pred))

    print(f"{train_or_validation.title()} Confusion Matrix:\n" + "-" * 30)
    print(confusion_matrix(y_data, y_pred))
