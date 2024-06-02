from csv import writer

import pandas as pd
import os

from evaluate.evaluate import save_status

original_data_file_name = 'datasets/adult/adult.data'
original_data = pd.read_csv(original_data_file_name, sep=",")

test_data = pd.read_csv('datasets/adult/adult.test', sep=",")

# Specify quasi-identifier and (single) sensitive attribute
QI = [1, 3, 6, 13]
QI_nominal = [1, 3, 6, 13]
QI_ordinal = []

# age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income

intervals = [0]
categorical = [1, 3, 5, 7, 6, 8, 9, 13, 14]

SA = 14  # Sensitive attribute: income

metrics = pd.DataFrame(columns=['num_suppressed_records',
                                'num_records',
                                'max_group_size',
                                'avg_group_size',
                                'k',
                                'l',
                                't',
                                'perc_recs',
                                'c_avg',
                                'bagging_acc',
                                'random_forest_acc',
                                'boosting_acc',
                                'bagging_auc',
                                'random_forest_auc',
                                'boosting_auc',
                                'bagging_prec',
                                'random_forest_prec',
                                'boosting_prec',
                                'bagging_recl',
                                'random_forest_recl',
                                'boosting_recl',
                                'bagging_f1',
                                'random_forest_f1',
                                'boosting_f1'])

file_name = 'anonymized/experiment_0/metrics.csv'

if not os.path.exists(file_name):
    metrics.to_csv(file_name)

# baseline
row_name = 'k: -, baseline'

roc_file_name = './anonymized/experiment_0/roc_adult_l=2_k=-'

save_status(original_data, original_data, test_data, QI, QI_nominal, QI_ordinal, SA, intervals, categorical,
            row_name, roc_file_name, file_name, no_suppression=True)

# baseline anonymization
for k in [2, 5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]:
    row_name = 'k: ' + str(k) + ', baseline'
    roc_file_name = './anonymized/experiment_0/roc_adult_l=2_k=' + str(k)
    train_data_file_name = ('./anonymized/experiment_0/baseline/anonymized_l=2_k=' + str(k) + '.csv')
    train_data = pd.read_csv(train_data_file_name, sep=";")

    save_status(original_data, train_data, test_data, QI, QI_nominal, QI_ordinal, SA, intervals, categorical,
                row_name, roc_file_name, file_name, no_suppression=False)

# own VGHs
for k in [ 15, 20, 25, 30, 50, 100, 150, 200]:

    for clustering in ['kmeans', 'agglomerative']:

        for emb in ['BERT', 'word2vec', 'average_word_embeddings_glove.6B.300d', 'msmarco-bert-base-dot-v5',
                    'multi-qa-mpnet-base-dot-v1', 'text-embedding-3-large', 'text-embedding-3-small',
                    'mistral-embed', 'jinaai-jina-embeddings-v2-base-en', 'fasttext',
                    'average_word_embeddings_komninos', 'average_word_embeddings_levy_dependency',
                    'average_word_embeddings_glove.840B.300d']:
            train_data_file_name = ('./anonymized/experiment_0/' + clustering + '/' + emb + '/adult_' + clustering
                                    + '_' + emb + '_anonymized_l=2_k=' + str(k) + '.csv')
            train_data = pd.read_csv(train_data_file_name, sep=";")

            row_name = 'k: ' + str(k) + ', clustering: ' + clustering + ',  embedding: ' + emb

            roc_file_name = (
                    './anonymized/experiment_0/' + clustering + '/' + emb + '/roc_adult_' + clustering + '_' + emb
                    + '_anonymized_l=2_k=' + str(k))

            save_status(original_data, train_data, test_data, QI, QI_nominal, QI_ordinal, SA, intervals, categorical,
                        row_name, roc_file_name, file_name, no_suppression=False)
