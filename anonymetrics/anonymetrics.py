import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def calculate_k_anonymity(df: pd.DataFrame, qa_indices: List[int]) -> int:
    """
    Calculates the k-anonymity of a datasets based on its quasi-identifiers (QI).

    Parameters
    -----------
    df : pandas DataFrame
        The input DataFrame containing the datasets.
    qa_indices : list of int
        A list of the indices of the QI columns in the DataFrame.

    Returns
    -----------
    k
        The minimum group size across all equivalence classes based on given QI.
    """

    # Group the DataFrame by the quasi-identifiers
    groupby_cols = df.columns[qa_indices].tolist()
    groupby_obj = df.groupby(groupby_cols)

    # Count number of records in each group
    group_counts = groupby_obj.size().reset_index(name='count')

    # Calculate k in k-anonymity
    k = group_counts['count'].min()

    return k


def calculate_l_diversity(df: pd.DataFrame, qa_indices: List[int], sa_indices: List[int]) -> int:
    """
    Calculates the distinct l-diversity of a datasets based on its quasi-identifiers (QI) and sensitive attributes (SA).

    Parameters
    -----------
    df : pandas DataFrame
        The input DataFrame containing both the QI and SA columns.
    qa_indices : list of int
        A list of the indices of the QI columns in the DataFrame.
    sa_indices : list of int
        A list of the indices of the SA columns in the DataFrame.

    Returns
    -----------
    l
        The minimum number of distinct values of each sensitive attribute across all equivalence classes given QI.
    """

    # Group the DataFrame by the quasi-identifiers
    groupby_cols = df.columns[qa_indices].tolist()
    groupby_obj = df.groupby(groupby_cols)

    # Calculate l in l-diversity

    grouped_df = groupby_obj[df.columns[sa_indices]]
    distinct_values = grouped_df.nunique()
    l = distinct_values.min().min()

    return l


def get_groups(df: pd.DataFrame, qa_indices: List[int]) -> List[pd.DataFrame]:
    """
    Extracts groups (equivalence classes) from a DataFrame based on some quasi-identifying factors.

    Parameters
    -----------
    df : pandas.DataFrame
        The DataFrame to extract equivalence classes from.
    qa_indices : list of int
        A list of column names to group by.

    Returns
    -----------
    groups
        list of pandas.DataFrame: A list of new DataFrame including disjunct groups.
    """

    # Group the DataFrame by the quasi identifiers
    groupby_col = df.columns[qa_indices].tolist()
    grouped = df.groupby(groupby_col)

    # Get the list of equivalence classes as DataFrames
    groups = [group for _, group in grouped]

    return groups


def get_group_lengths(groups: List) -> List:
    return [len(group) for group in groups]


def calculate_sensitive_attr_prob_dist(df: pd.DataFrame, sa_index: int):
    """
    Calculates the probability distribution of the sensitive attribute in a datasets.

    Parameters
    -----------
    df : pandas.DataFrame
        The datasets containing the sensitive attribute.
    sa_indices : list of int
        A list of the indices of the SA columns in the DataFrame.

    Returns
    -----------
    dist
        A tuple containing the probability distributions of the
        sensitive attribute in the datasets, respectively.
    """

    dist = df.iloc[:, sa_index].value_counts(normalize=True)
    return dist


# see https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf, S.6
def emd_numerical(dist_0: pd.Series, dist_1: pd.Series) -> float:
    """
    Calculates the (ordered) earth mover distance (emd) between two probability distributions when the attribute value
    is numerical.

    Parameters
    -----------
    dist_0 : pandas.Series
        A probability distribution represented as a pandas Series. The index should correspond to attribute values.
    dist_1 : pandas.Series
        Another probability distribution represented as a pandas Series. The index should correspond to attribute values.

    Returns
    -----------
    emd
        The earth mover distance between the two input distributions.
    """
    m = dist_0.size  # number of attribute values

    # convert tuple/list of numbers to their mean value
    dist_0.index = dist_0.index.map(lambda x: np.mean(x))
    dist_1.index = dist_1.index.map(lambda x: np.mean(x))

    P = dist_0.sort_index().array
    Q = dist_1.sort_index().array
    R = P - Q

    if m > 1:
        emd = (1 / (m - 1)) * sum([abs(sum(R[:(i + 1)])) for i in range(m)])
    else:
        emd = 0

    return emd


# see https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf, S.6
def emd_categorical(dist_0: pd.Series, dist_1: pd.Series) -> float:
    """
    Calculates the equal earth mover distance (emd) between two probability distributions when the attribute value
    is categorical.

    Parameters
    -----------
    dist_0 : pandas.Series
        A probability distribution represented as a pandas Series. The index should correspond to attribute values.
    dist_1 : pandas.Series
        Another probability distribution represented as a pandas Series. The index should correspond to attribute values.

    Returns
    -----------
    emd
        The earth mover distance between the two input distributions.
    """
    # Convert indices to
    dist_0.index = dist_0.index.astype(str)
    dist_1.index = dist_1.index.astype(str)

    P = dist_0.sort_index().array
    Q = dist_1.sort_index().array

    R = P - Q

    emd = 0.5 * sum(abs(R))

    return emd


def calculate_t_closeness(df: pd.DataFrame, qa_indices: List[int], sa_index: int) -> float:
    """
    Calculates the t-closeness measure for a sensitive attribute in a DataFrame.

    The t-closeness measure is a privacy metric that quantifies the degree to which a sensitive attribute can be inferred
    from quasi-identifying attributes in a datasets. Specifically, t-closeness requires that the distribution of sensitive
    attribute values within each group of records that share the same quasi-identifiers is "close" to the distribution of
    sensitive attribute values in the entire datasets.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    qa_indices : List[int]
        A list of column indices for the quasi-identifying attributes.
    sa_index : int
        The column index for the sensitive attribute.

    Returns
    ----------
    t
        The maximum EMD distance between the probability distribution of the sensitive attribute values within any group
        of records that share the same quasi-identifiers, and the probability distribution of the sensitive attribute
        values in the entire datasets. A smaller t-closeness value indicates a higher degree of privacy.
    """

    attr_value_type = df.iloc[:, sa_index].dtype.name

    groups = get_groups(df, qa_indices)
    dist_dataset = calculate_sensitive_attr_prob_dist(df, sa_index)

    t = 0

    # numerical attribute
    if attr_value_type in ["int64", "float32", "float32", "tuple"]:

        for group in groups:
            dist_group = calculate_sensitive_attr_prob_dist(group, sa_index)
            dist_group = pd.Series(data=dist_group, index=dist_dataset.index).fillna(0)
            t_new = emd_numerical(dist_group, dist_dataset)
            t = max(t, t_new)

    else:

        for group in groups:
            dist_group = calculate_sensitive_attr_prob_dist(group, sa_index)
            dist_group = pd.Series(data=dist_group, index=dist_dataset.index).fillna(0)
            t_new = emd_categorical(dist_group, dist_dataset)
            t = max(t, t_new)

    return t


def get_group_sizes(df: pd.DataFrame, qa_indices: List[int]) -> np.ndarray:
    """
    Extracts the group sizes based on the quasi-identifiers (QI) in a datasets.

    Parameters
    -----------
    df : pandas DataFrame
        The input DataFrame containing the datasets.
    qa_indices : list of int
        A list of the indices of the QI columns in the DataFrame.

    Returns
    -----------
    numpy.ndarray
        An array containing the group sizes.
    """
    groups = get_groups(df, qa_indices)
    lengths = get_group_lengths(groups)

    return lengths


def get_diversities(df: pd.DataFrame, qa_indices: List[int], sa_indices: List[int]) -> np.ndarray:
    """
    Extracts ...

    Parameters
    -----------
    df : pandas DataFrame
        The input DataFrame containing the datasets.
    qa_indices : list of int
        A list of the indices of the QI columns in the DataFrame.
    sa_indices : list of int
        A list of the indices of the sensitive attribute columns in the DataFrame.

    Returns
    -----------
    numpy.ndarray
        An array containing the diversities per group.
    """
    groups = get_groups(df, qa_indices)
    diversities = [calculate_l_diversity(group, qa_indices, sa_indices) for group in groups]

    return diversities


def get_closenesses(df: pd.DataFrame, qa_indices: List[int], sa_index: int) -> np.ndarray:
    """
    Extracts ...

    Parameters
    -----------
    df : pandas DataFrame
        The input DataFrame containing the datasets.
    qa_indices : list of int
        A list of the indices of the QI columns in the DataFrame.
    sa_index : list of int
        The index of a sensitive attribute

    Returns
    -----------
    numpy.ndarray
        An array containing the closeness per group.
    """

    groups = get_groups(df, qa_indices)
    dist_dataset = calculate_sensitive_attr_prob_dist(df, sa_index)

    closenesses = []

    for group in groups:
        dist_group = calculate_sensitive_attr_prob_dist(group, sa_index)
        dist_group = pd.Series(data=dist_group, index=dist_dataset.index).fillna(0)
        closenesses.append(emd_categorical(dist_group, dist_dataset))

    return closenesses


def get_count_per_group_size(df: pd.DataFrame, qa_indices: List[int]) -> np.ndarray:
    """
    Calculates the count per group size based on the quasi-identifiers (QI) in a datasets.

    Parameters
    -----------
    df : pandas DataFrame
        The input DataFrame containing the datasets.
    qa_indices : list of int
        A list of the indices of the QI columns in the DataFrame.

    Returns
    -----------
    numpy.ndarray
        An array containing the count per group size.
    """
    # Call the get_group_sizes function to calculate group sizes
    group_sizes = get_group_sizes(df, qa_indices)

    counter = Counter(group_sizes)

    keys = []
    values = []

    for group_size in counter:
        keys.append(group_size)
        values.append(counter[group_size])

    max = np.max(keys)

    counts = np.zeros(max + 2)

    for i in range(len(keys)):
        counts[keys[i]] = keys[i] * values[i]

    # Return the count per group size
    return counts


def plot_count_per_group_size(count_per_group_size: np.ndarray):
    """
    Plots the count per group size.

    Parameters
    -----------
    count_per_group_size : numpy.ndarray
        An array containing the count per group size.

    Returns
    -----------
    None
    """

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(count_per_group_size)), count_per_group_size)

    # Adding labels and title
    plt.xlabel('Group Sizes')
    plt.ylabel('Count')
    plt.title('Count per group size')

    # Display the plot
    plt.show()


def plot_group_sizes_histogram(group_sizes: np.ndarray):
    """
    Plots a histogram of group sizes.

    Parameters
    -----------
    group_sizes : numpy.ndarray
        An array containing the group sizes.
    """
    plt.hist(group_sizes, bins='auto')
    plt.xlabel('group size')
    plt.ylabel('frequency')
    plt.title('histogram of group sizes')
    plt.show()


def plot_l_diversities_histogram(l_diversities: np.ndarray):
    """
    Plots a histogram of diversities.

    Parameters
    -----------
    l_diversities : np.ndarray
        ...
    """
    plt.hist(l_diversities, bins='auto')
    plt.xlabel('$l$-diversity')
    plt.ylabel('frequency')
    plt.title('histogram of diversities')
    plt.show()


def plot_t_closenesses_histogram(t_closenesses: np.ndarray):
    """
    Plots a histogram of closenesses.

    Parameters
    -----------
    t_closenesses : np.ndarray
        ...
    """
    plt.hist(t_closenesses, bins='auto')
    plt.xlabel('$t$-closeness')
    plt.ylabel('frequency')
    plt.title('histogram of closenesses')
    plt.show()
