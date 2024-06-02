import pandas as pd
import time
import os
from preprocess.vectorize import Vectorize
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE

from cluster_attribute import get_kmeans_cluster, get_agglomerative_cluster, get_categories
from preprocess.preprocess import preprocess_string, post_process_vectors

dataset_name = "adult"


def plot_embeddings(dataset_name, embedding_name, column_name, vectors, raw_values):
    raw_values = [string.lstrip() for string in raw_values]

    folder_name = "embedding_pics/adult/" + column_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" has been created successfully.')
    else:
        print(f'Folder "{folder_name}" already exists.')

    # TSNE scatter plot
    tsne = TSNE(n_components=2, random_state=0, perplexity=2).fit_transform(vectors)

    ax = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=raw_values, legend=False)
    for i, txt in enumerate(raw_values):
        ax.annotate(txt, (tsne[i, 0], tsne[i, 1]), fontsize=12, rotation=0, ha='center', va='bottom')

    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.tight_layout()
    file_name = time.strftime(
        folder_name + "/" + dataset_name + "_" + emb + "_"
        + column_name + "_tsne_scatter_plot_%d-%m-%y-%H-%M.png")
    plt.savefig(file_name, dpi=300)

    # UMAP scatter plot

    umap_model = umap.UMAP(n_components=2, random_state=0, n_neighbors=5, min_dist=0.1)
    umap_vectors = umap_model.fit_transform(vectors)

    plt.figure()

    ax = sns.scatterplot(x=umap_vectors[:, 0], y=umap_vectors[:, 1], hue=raw_values, legend=False)
    for i, txt in enumerate(raw_values):
        ax.annotate(txt, (umap_vectors[i, 0], umap_vectors[i, 1]), fontsize=12, rotation=0, ha='center', va='bottom')

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    file_name = time.strftime(
        folder_name + "/" + dataset_name + "_" + embedding_name + "_"
        + column_name + "_umap_scatter_plot_%d-%m-%y-%H-%M.png")
    plt.savefig(file_name, dpi=300)


def plot_dendrogram(file_name, model, values):
    plt.figure()
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, labels=values, truncate_mode=None, p=3, orientation='right')
    plt.xlabel("distance between two merged clusters $C_j, C_k$")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def create_vgh(dataset_name: str, vectors, values, raw_values, clustering_name: str, embedding_name: str,
               column_name: str):
    """
    This function creates a value generalization hierarchy (VGH) for a specified column of a pandas DataFrame (T),
    it then stores the result as a csv file in a specified folder.

    Arguments:
        dataset_name (str) : The name of the datasets, used for naming the output folder and file.
        vectors : List of word embeddings for pre-processed values.
        values : List of pre-processed values of the column.
        raw_values : List of original values.
        clustering_name : Clustering technique.
        embedding_name : Pre-trained embeddings.
        column_name : The name of the column in the dataframe.

    Returns:
        vgh (list): A (ordered) list of lists representing the value generalization hierarchy of the specified column.

    """

    # CLUSTERING

    cluster_centers = vectors

    num_of_values = len(raw_values)

    df = pd.DataFrame(index=range(num_of_values),
                      columns=range(num_of_values))  # dataframe size: num_of_values^2
    df.iloc[:, 0] = raw_values

    labels = list(range(num_of_values))

    i = 1
    while max(labels) > 0:

        if clustering_name == 'kmeans':
            labels, cluster_centers = get_kmeans_cluster(labels, len(values) - i, vectors=cluster_centers)

        elif clustering_name == 'agglomerative':
            categories = get_categories(values, labels)
            y_labels_in_dendrogram = [raw_values[categories.index(val)] for val in categories]
            labels, model = get_agglomerative_cluster(len(values) - i, vectors=cluster_centers)
            if i == 1:
                file_name = time.strftime(
                    folder_name + "/" + dataset_name + "_" + clustering_name + "_" + embedding_name + "_"
                    + column_name + "_dendrogram_%d-%m-%y-%H-%M.png")
                plot_dendrogram(file_name, model, y_labels_in_dendrogram)

        categories = get_categories(values, labels)

        vgh.append(categories)

        col = []
        for value in values:
            for cat in categories:
                if value in cat:
                    if len(cat) == num_of_values:
                        entry = '*'
                    else:
                        raw_cat = []
                        for value in cat:
                            index = values.index(value)
                            entry = raw_values[index]
                            entry = entry[entry.find(entry.strip()):]
                            raw_cat.append(entry)
                        entry = str(set(raw_cat)).replace('\'', '')

                    col.append(entry)

        df.iloc[:, i] = col

        i += 1

    # STORE RESULTS

    file_name = time.strftime(folder_name + "/" + dataset_name + "_" + clustering_name + "_" + embedding_name + "_"
                              + column_name + "_vgh_%d-%m-%y-%H-%M.csv")

    print(file_name)

    df.to_csv(file_name, index=False, header=False)

    with open(file_name, 'r') as file:
        lines = file.readlines()
    lines = [line.replace('"', '') for line in lines]
    lines = [line.replace('{', '"{') for line in lines]
    lines = [line.replace('}', '}"') for line in lines]
    lines = [line.lstrip() for line in lines]
    with open(file_name, 'w') as file:
        file.writelines(lines)

    return vgh


if __name__ == '__main__':

    start_cpu_time = time.process_time()

    T = pd.read_csv("./datasets/adult/adult.data.csv")

    # workclass, education, marital-status, native-country, occupation, race, income, sex
    attributes = [13, 1, 3, 5, 13, 6, 8, 14, 9]

    # keys needed for: 'text-embedding-3-large', 'text-embedding-3-small', 'mistral-embed'
    for emb in ['word2vec', 'BERT', 'word2vec', 'average_word_embeddings_glove.6B.300d', 'msmarco-bert-base-dot-v5',
                'multi-qa-mpnet-base-dot-v1', 'jinaai/jina-embeddings-v2-base-en', 'fasttext',
                'average_word_embeddings_komninos', 'average_word_embeddings_levy_dependency',
                'average_word_embeddings_glove.840B.300d']:

        embedding = Vectorize(emb)

        # rename to store results without /
        if emb == 'jinaai/jina-embeddings-v2-base-en':
            emb = 'jinaai-jina-embeddings-v2-base-en'

        # create embeddings only once

        list_of_values = []
        list_of_raw_values = []
        list_of_vectors = []

        for i in attributes:

            column_name = T.columns[i]

            raw_values = list(set(T[column_name]))
            num_of_values = len(raw_values)

            values = [preprocess_string(value) for value in raw_values]

            vgh = [values]

            labels = list(range(num_of_values))
            vectors = post_process_vectors(embedding, values)  # in first step vectors = cluster_centers

            list_of_values.append(values)
            list_of_raw_values.append(raw_values)
            list_of_vectors.append(vectors)

            # PLOT TSNE AND UMAP EMBEDDINGS

            folder_name = "embedding_pics/adult/" + column_name
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f'Folder "{folder_name}" has been created successfully.')
            else:
                print(f'Folder "{folder_name}" already exists.')

            # not needed for 2 attribute values
            if i not in [14, 9]:
                plot_embeddings(dataset_name, emb, column_name, vectors, raw_values)

        for clustering in ['kmeans','agglomerative', 'kmeans']:

            # PREPARE FOLDER

            folder_name = "anonymized/adult/" + clustering + "/" + emb

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f'Folder "{folder_name}" has been created successfully.')
            else:
                print(f'Folder "{folder_name}" already exists.')

            # CREATE VGH FOR EACH ATTRIBUTE

            j = 0
            for i in attributes:

                vectors = list_of_vectors[j]
                values = list_of_values[j]
                raw_values = list_of_raw_values[j]

                column_name = T.columns[i]

                create_vgh(dataset_name=dataset_name, vectors=vectors, values=values, raw_values=raw_values,
                           clustering_name=clustering, embedding_name=emb, column_name=column_name)

                j += 1

    end_cpu_time = time.process_time()
    elapsed_cpu_time = end_cpu_time - start_cpu_time
    print("execution time (s): ")
    print(elapsed_cpu_time)
