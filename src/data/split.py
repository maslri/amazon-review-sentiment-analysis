"""
Utilities for dataset splitting.
"""

from sklearn.model_selection import train_test_split


def create_train_validation_split(
    df,
    target_column="overall",
    test_size=0.2,
    random_state=42,
):
    """
    Split a dataframe into train and validation sets while
    preserving the target distribution.

    Parameters
    ----------
    df : pandas.DataFrame

    target_column : str
        Target column.

    test_size : float
        Validation ratio.

    random_state : int
        Random seed.

    Returns
    -------
    train_df : pandas.DataFrame

    valid_df : pandas.DataFrame
    """

    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
    )


def class_distribution(df, target_column="overall"):
    """
    Calculate percentage distribution of target classes.
    """

    return df[target_column].value_counts(normalize=True).sort_index().mul(100).round(2)
