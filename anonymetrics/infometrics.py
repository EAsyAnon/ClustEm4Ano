import numpy as np
import pandas as pd
from collections import Counter


def entropy_info_loss(df1: pd.DataFrame, df2: pd.DataFrame, column_index: int) -> float:
    """
    Calculate the information loss of the anonymized dataframe `df2` using conditional entropy.

    Parameters
    -----------
    df1 : pd.DataFrame
        Original, non-anonymized dataframe.
    df2 : pd.DataFrame
        Anonymized dataframe.
    column_index : int
        Index of the column for which to calculate the information loss.

    Returns
    -----------
    info_loss (float): Information loss due to anonymization.
    """
    # Check if the dataframes are of the same length
    if len(df1) != len(df2):
        return 0

    # Compute the frequencies for each attribute value in the original datasets
    original_values = df1.iloc[:, column_index].values
    original_value_counts = Counter(original_values)

    # Compute the probabilities for each attribute value in the original datasets
    original_prob = {key: val / len(df1) for key, val in original_value_counts.items()}

    # Compute the probabilities for each anonymized attribute value
    anonymized_values = df2.iloc[:, column_index].values
    anonymized_value_counts = Counter(anonymized_values)
    anonymized_prob = {key: val / len(df2) for key, val in anonymized_value_counts.items()}

    # Compute the entropy-based information loss for each anonymized row
    info_loss = 0
    for i in range(len(df1)):
        anonymized_value = anonymized_values[i]
        # If anonymized value in row i is a frozenset, calculate the conditional probabilities, otherwise: add 0
        if isinstance(anonymized_value, frozenset):
            conditional_probs = [original_prob.get(val, 0)/anonymized_prob.get(anonymized_value, 0) for val in anonymized_value]
            info_loss += np.sum([-p * np.log2(p) for p in conditional_probs if p > 0])

    return info_loss


def numerical_info_loss(df1: pd.DataFrame, df2: pd.DataFrame, index):
    """
    This function calculates the information loss between the original datasets (df1) and
    the anonymized datasets (df2) based on a specified column index.

    Assumes df1 and df2 have the same number of records.

    The information loss is calculated using the formula:
        Π(D,g(D)) := (1/n*r) * Σ (upper_ij - lower_ij) / (max_j - min_j)
    where:
        upper_ij and lower_ij are the upper and lower bounds of the generalized attribute value interval,
        min_j and max_j are the minimum and maximum attribute value before generalization.
    Parameters
    -----------
    df1 : pd.DataFrame
        The original, non-anonymized tabular data.
    df2 : pd.DataFrame
        The anonymized tabular data. Each cell should contain a tuple representing the generalized interval.
    index : int or list of int
        The column index or indices based on which the information loss will be calculated.

    Returns
    -----------
    info_loss
        A numerical value representing the information loss between the original data and the anonymized data.
    """
    n = df2.shape[0]

    if not n == df1.shape[0]:
        return 0

    r = 1

    total_info_loss = 0.0

    for idx in np.atleast_1d(index):

        # assumes, that if first record at idx is tuple, then all records at index idx are tuples

        # print(df1.iloc[:, idx])

        if isinstance(df1.iloc[0, idx], tuple):
            min_j = df1.iloc[:, idx].apply(lambda x: x[0]).min()
            max_j = df1.iloc[:, idx].apply(lambda x: x[1]).max()
        else:
            min_j = df1.iloc[:, idx].min()
            max_j = df1.iloc[:, idx].max()

        for i in range(n):

            if isinstance(df2.iloc[i, idx], tuple):
                lower_ij, upper_ij = df2.iloc[i, idx]
            else:
                lower_ij = df2.iloc[i, idx].min()
                upper_ij = df2.iloc[i, idx].max()

            if isinstance(df1.iloc[i, idx], tuple):
                lower_ij_1, upper_ij_1 = df1.iloc[i, idx]
            else:
                lower_ij_1 = df1.iloc[i, idx].min()
                upper_ij_1 = df1.iloc[i, idx].max()

            if lower_ij_1 == lower_ij and upper_ij_1 == upper_ij:
                loss = 0
            else:
                loss = (upper_ij - lower_ij) / (max_j - min_j)

            total_info_loss += loss

    info_loss = total_info_loss / (n * r)
    return info_loss


def euclid_info_loss(df1: pd.DataFrame, df2: pd.DataFrame, index):
    """
    Calculates the information loss between the original datasets (df1) and
    the anonymized datasets (df2) based on the Euclidean distance.

    The information loss is calculated using the formula:
        loss = (1/n) * Σ_i sqrt(Σ (x_ij - y_ij)^2),
    where:
        x_ij and y_ij are the original and anonymized attribute values, respectively.
        The summation inside the sqrt is over all attributes in index, and the outer summation is over all records.

    Parameters
    -----------
    df1 : pd.DataFrame
        The original, non-anonymized tabular data.
    df2 : pd.DataFrame
        The anonymized tabular data.
    index : int or list of int
        The column index or indices based on which the information loss will be calculated.

    Returns
    -----------
    info_loss
        A numerical value representing the information loss between the original data and the anonymized data.
    """

    if len(df1) != len(df2):
        return 0

    n = df1.shape[0]  # number of records

    total_info_loss = 0.0

    for i in range(n):

        for idx in np.atleast_1d(index):

            sum_of_squares = 0.0

            # handle tuples in by taking their mean
            if isinstance(df2.iloc[i, idx], tuple):
                df1_val = np.mean(df1.iloc[i, idx])
            else:
                df1_val = df1.iloc[i, idx]

            if isinstance(df2.iloc[i, idx], tuple):
                df2_val = np.mean(df2.iloc[i, idx])
            else:
                df2_val = df2.iloc[i, idx]

            # print(df1_val, df2_val)

            sum_of_squares += (df1_val - df2_val) ** 2

        dist = np.sqrt(sum_of_squares)

        total_info_loss += dist

    info_loss = total_info_loss / n
    return info_loss
