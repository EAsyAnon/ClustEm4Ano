import numpy as np
from csv import writer
from typing import List, Any
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time

from anonymetrics.anonymetrics import get_groups, calculate_l_diversity, calculate_t_closeness, \
    calculate_k_anonymity
from anonymetrics.utilitymetrics import c_avg


def preprocess(df: pd.DataFrame, intervals, categorical) -> tuple[DataFrame | Any, Series]:
    """
    Preprocess by performing one-hot encoding on categorical attributes.

    Parameters:
        df (pd.DataFrame): The input dataframe to be preprocessed.

    Returns:
        pd.DataFrame, pd.Series: A tuple of two items—a preprocessed dataframe and a new label.
    """

    df = remove_intervals(df, intervals)

    df = one_hot_encoding(df, categorical)

    new_label = df.iloc[:, -1]
    df = df.drop(df.columns[-2:], axis=1)

    return df, new_label


def save_status(previous_data: pd.DataFrame, anonymized_data: pd.DataFrame, test_data: pd.DataFrame,
                qa_indices: List[int], qa_nominal: List[int], qa_ordinal: List[int], sa_index: int,
                intervals: List[int], categorical: List[int], row_name: str, roc_file_name: str,
                file_name: str, no_suppression: False):
    """
    Saves privacy, utility, and machine learning (Random Forest, Decision Tree, Bagging, Boosting) downstream efficacy
    metrics in csv file.

    Args:
        previous_data (pd.DataFrame): Data before anonymization
        anonymized_data (pd.DataFrame): Anonymized data
        test_data (pd.DataFrame): Test data for machine learning evaluation
        qa_indices (List[int]): List of indices corresponding to quasi-identifiers
        qa_nominal (List[int]): List of indices corresponding to nominal quasi-identifiers
        qa_ordinal (List[int]): List of indices corresponding to ordinal quasi-identifiers
        sa_index (int): Index corresponding to the sensitive attribute
        file_name (str): Name of the file to save results
        row_name (str): Names the row in the metrics.csv
        roc_file_name (str): Name of the file to save ROC curves
        no_suppression (bool): If True, then no suppression was performed (e.g., baseline)

    Returns:
        None
    """

    groups = get_groups(anonymized_data, qa_indices)
    lengths = [len(group) for group in groups]

    if not no_suppression:
        _num_suppressed_records = np.max(lengths)
    else:
        _num_suppressed_records = 0

    print('Number of suppressed records ', _num_suppressed_records)

    _num_records = len(anonymized_data) - _num_suppressed_records
    print('Number of records ', _num_records)

    if not no_suppression:
        group_lengths_filtered = [l for l in lengths if l != _num_suppressed_records]
    else:
        group_lengths_filtered = lengths

    # print('Minimal equivalence class size ', np.min(group_lengths_filtered))

    _max_group_size = np.max(group_lengths_filtered)
    print('Maximal equivalence class size ', _max_group_size)

    _avg_group_size = np.mean(group_lengths_filtered)
    print('Average equivalence class size ', _avg_group_size)

    _num_groups = len(group_lengths_filtered)
    print('Number of equivalence classes ', _num_groups)

    if not no_suppression:
        non_max_length_groups = [group for group in groups if len(group) != _num_suppressed_records]
        anonymized_data_without_suppressed = pd.concat(non_max_length_groups, axis=0, ignore_index=True)
    else:
        non_max_length_groups = groups
        anonymized_data_without_suppressed = anonymized_data

    _k = calculate_k_anonymity(anonymized_data_without_suppressed, qa_indices)
    print('k ', _k)
    _l = calculate_l_diversity(anonymized_data_without_suppressed, qa_indices, [sa_index])
    print('l ', _l)
    _t = calculate_t_closeness(anonymized_data_without_suppressed, qa_indices, sa_index)
    print('t ', _t)

    _c_avg = c_avg(anonymized_data_without_suppressed, non_max_length_groups, _k)
    print('c_avg ', _c_avg)

    _perc_recs = len(anonymized_data_without_suppressed) / len(anonymized_data)
    print("Percentage of records retained ", _perc_recs)

    # ML utility

    anonymized_data = anonymized_data.drop(anonymized_data.columns[[0, 2, 4, 5, 7, 8, 9]], axis=1, inplace=False)

    test_data = test_data.drop(test_data.columns[[0, 2, 4, 5, 7, 8, 9]], axis=1, inplace=False)

    data = pd.concat([anonymized_data, test_data], ignore_index=True)
    preprocessed = preprocess(data, [], [0,1,2,6,7])

    train_data = preprocessed[0][:anonymized_data.shape[0]]
    train_label = preprocessed[1][:anonymized_data.shape[0]]

    test_data = preprocessed[0][anonymized_data.shape[0]:]
    test_label = preprocessed[1][anonymized_data.shape[0]:]

    # ML

    # Random Forest

    print('RANDOM FOREST')

    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    clf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3, error_score='raise')
    grid_search.fit(train_data, train_label)

    best_clf = grid_search.best_estimator_
    _random_forest_accuracy = best_clf.score(test_data, test_label)
    print(_random_forest_accuracy)

    pred = best_clf.predict(test_data)

    _random_forest_prec = metrics.precision_score(test_label, pred)
    _random_forest_recl = metrics.recall_score(test_label, pred)
    _random_forest_f1 = metrics.f1_score(test_label, pred)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred, pos_label=1)
    _random_forest_auc = metrics.auc(fpr, tpr)
    print('auc', _random_forest_auc)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {_random_forest_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    file_name_roc = time.strftime(
        roc_file_name + "_random_forest_roc_%d-%m-%y-%H-%M.png")
    # plt.savefig(file_name_roc, dpi=300)

    # Bagging

    print('BAGGING')

    base_estimator = DecisionTreeClassifier()

    param_grid = {
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__max_depth': [4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150],
        'n_estimators': [10, 50]
    }

    clf = BaggingClassifier(estimator=base_estimator)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(train_data, train_label)

    best_clf = grid_search.best_estimator_
    _bagging_accuracy = best_clf.score(test_data, test_label)
    print(_bagging_accuracy)

    pred = best_clf.predict(test_data)

    _bagging_prec = metrics.precision_score(test_label, pred)
    _bagging_recl = metrics.recall_score(test_label, pred)
    _bagging_f1 = metrics.f1_score(test_label, pred)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred, pos_label=1)
    _bagging_auc = metrics.auc(fpr, tpr)
    print('auc', _random_forest_auc)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {_bagging_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    file_name_roc = time.strftime(
        roc_file_name + "_bagging_roc_%d-%m-%y-%H-%M.png")
    # plt.savefig(file_name_roc, dpi=300)

    # Boosting

    print('BOOSTING')

    base_estimator = DecisionTreeClassifier()

    param_grid = {
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__max_depth': [4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150],
        'n_estimators': [10, 50]
    }

    clf = AdaBoostClassifier(estimator=base_estimator)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(train_data, train_label)

    best_clf = grid_search.best_estimator_
    _boosting_accuracy = best_clf.score(test_data, test_label)
    print(_boosting_accuracy)

    pred = best_clf.predict(test_data)

    _boosting_prec = metrics.precision_score(test_label, pred)
    _boosting_recl = metrics.recall_score(test_label, pred)
    _boosting_f1 = metrics.f1_score(test_label, pred)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred, pos_label=1)
    _boosting_auc = metrics.auc(fpr, tpr)
    print('auc', _boosting_auc)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {_boosting_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    file_name_roc = time.strftime(
        roc_file_name + "_boosting_roc_%d-%m-%y-%H-%M.png")
    # plt.savefig(file_name_roc, dpi=300)

    # Create a dictionary with the values
    new_row = {
        'num_suppressed_records': _num_suppressed_records,
        'num_records': _num_records,
        'max_group_size': _max_group_size,
        'avg_group_size': _avg_group_size,
        'k': _k,  # k anonymity
        'l': _l,  # l diversity
        't': _t,  # t closeness
        'perc_recs': _perc_recs,  # indicates suppression
        'c_avg': _c_avg,
        # accuracy
        'bagging_acc': _bagging_accuracy,
        'random_forest_acc': _random_forest_accuracy,
        'boosting_acc': _boosting_accuracy,
        # auc
        'bagging_auc': _bagging_auc,
        'random_forest_auc': _random_forest_auc,
        'boosting_auc': _boosting_auc,
        'bagging_prec': _bagging_auc,
        'random_forest_prec': _random_forest_prec,
        'boosting_prec': _boosting_prec,
        'bagging_recl': _bagging_recl,
        'random_forest_recl': _random_forest_recl,
        'boosting_recl': _boosting_recl,
        'bagging_f1':  _bagging_auc,
        'random_forest_f1': _random_forest_f1,
        'boosting_f1': _boosting_f1
    }

    print(new_row)

    with open(file_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerow([row_name, _num_suppressed_records, _num_records, _max_group_size, _avg_group_size, _k, _l, _t,
                                _perc_recs, _c_avg,
                                _bagging_accuracy, _random_forest_accuracy, _boosting_accuracy,
                                _bagging_auc, _random_forest_auc, _boosting_auc,
                                _bagging_prec, _random_forest_prec, _boosting_prec,
                                _bagging_recl, _random_forest_recl, _boosting_recl,
                                _bagging_f1, _random_forest_f1, _boosting_f1])
        f.close()

import pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def one_hot_encoding(df: pd.DataFrame, categorical) -> pd.DataFrame:
    """
    Perform one-hot encoding on a DataFrame for specified categorical columns.

    Parameters:
        df (pandas.DataFrame): The input DataFrame including categorical columns to be one-hot encoded.

    Returns:
        pandas.DataFrame: A new DataFrame with one-hot encoded columns for each unique value in the specified
        categorical columns.

    """

    categorical_columns = [df.columns[i] for i in categorical]

    final_df = df

    for col in categorical_columns:

        # Convert the column values to lists (if not already in list format)
        df[col] = df[col].apply(lambda x: [cls.strip() for cls in x.strip('{}').split(',')] if isinstance(x, str) and x.startswith('{') else x)
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, int) else x)
        df[col] = df[col].apply(lambda x: list(x) if isinstance(x, frozenset) else x)

        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])

        x = df[col].tolist()

        mlb = MultiLabelBinarizer()

        encoded_genres = mlb.fit_transform(x)

        class_labels = mlb.classes_

        encoded_df = pd.DataFrame(encoded_genres, columns=class_labels)
        final_df = pd.concat([final_df, encoded_df], axis=1)

        final_df.drop([col], axis=1, inplace=True)

    return final_df

from typing import List

import pandas as pd


def is_tuple(value):
    return isinstance(value, tuple)


def remove_intervals(df: pd.DataFrame, intervals: List[int]) -> pd.DataFrame:
    """
    Remove interval values from tuple columns in a DataFrame and replace them with the mean of min and max.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing columns with interval values of type 'tuple'.

    Returns:
        pd.DataFrame: A new DataFrame with interval values replaced by their mean.
    """

    for j in intervals:

        my_column = df.iloc[:, j]
        new_column = []

        i = 0
        for val in my_column:
            if isinstance(val, str):
                if "*" in val:
                    val = 49.5
                elif "-" in val:
                    left, right = val.split("-")

                    left_value = int(left)
                    right_value = int(right)

                    val = (left_value + right_value) / 2
                else:
                    val = int(val)

            new_column.append(val)
            i+=1

        df.iloc[:, j] = new_column

        interval_columns = [col for col in df.columns.tolist() if df[col].apply(lambda x: is_tuple(x)).any()]

        # Compute mean of min and max for tuple columns and replace interval by mean
        for col in interval_columns:
            df[col] = df[col].apply(lambda x: (min(x) + max(x)) / 2 if isinstance(x, tuple) else x)

    return df
