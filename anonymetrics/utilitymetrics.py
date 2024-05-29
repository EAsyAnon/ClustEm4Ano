from typing import List
import pandas as pd


def avg_size(frames: List[pd.DataFrame]) -> float:
    """
    Calculates the average size of a list of dataframes.

    Parameters
    ----------
    frames : list of pd.DataFrame
        List of dataframes.

    Returns
    -------
    float
        The average size of the frame sizes.
    """

    num_of_entries = 0
    for frame in frames:
        num_of_entries += len(frame)

    return num_of_entries / len(frames)


def c_avg(df: pd.DataFrame, groups: List[pd.DataFrame], k: int) -> float:
    """
    Calculates the average group size metric.

    Parameters
    ----------
    df : pd.DataFrame
        The entire dataframe.
    groups : list of pd.DataFrame
        List of groups in the dataframe.
    k : int
        Pre-defined k-anonymity of the datasets.

    Returns
    -------
    float
        The average equivalence class size metric.
    """
    return len(df) / (len(groups) * k)


def c_dm(groups: List[pd.DataFrame]) -> int:
    """
    Calculates the discernibility metric.

    Parameters
    ----------
    groups : list of pd.DataFrame
        List of groups in the dataframe.

    Returns
    -------
    int
        The discernibility metric.
    """
    return sum(len(group) ** 2 for group in groups)
