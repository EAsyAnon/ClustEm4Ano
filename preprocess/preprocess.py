import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from preprocess.vectorize import Vectorize

def preprocess_string(value):

    # Convert letters to lowercase
    value = value.lower()
    # Replace '-' and '_' with empty space
    value = value.replace('-', ' ')
    value = value.replace('_', ' ')
    # Tokenize text
    tokens = nltk.word_tokenize(value)
    # Remove numbers
    tokens = [word for word in tokens if not word.isdigit()]
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    preprocessed = ' '.join(tokens)
    if preprocessed == '':
        return str("none")
    else:
        return preprocessed


def preprocess_dataframe(df: pd.DataFrame):
    """
     Preprocesses string values in a Pandas DataFrame for NLP.

     Parameters:
     -----------
     df : pandas.DataFrame
         Input DataFrame.
     columns : List[int]
         Columns where values should be preprocessed.

     Returns:
     --------
     pandas.DataFrame
         DataFrame with string columns preprocessed for NLP:
         - Converts letters to lowercase
         - Tokenizes text
         - Removes numbers
         - Removes punctuation
         - Removes stop words
    """

    df_raw = df.copy()

    # Apply preprocessing function to all string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: preprocess_string(x) if pd.notnull(x) else x)

    return df, df_raw


def post_process_vectors(embedding: Vectorize, vectors):

    if embedding.embedding in ['BERT']:
        vectors = np.stack([embedding.vectorize(ws).numpy() for ws in vectors])
    else:
        vectors = np.stack([embedding.vectorize(ws) for ws in vectors])
    return vectors
