import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

import os

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_THREADING_LAYER'] = "TBB"
import numpy as np

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import gensim.downloader

import fasttext.util

from openai import OpenAI
from mistralai.client import MistralClient

import nltk

nltk.download('punkt')
nltk.download('stopwords')

import torch
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

# Hugging Face
os.environ['HF_TOKEN'] = "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"  # "<your token here>"


class Vectorize:
    def __init__(self, embedding):
        self.embedding = embedding
        self.sentence_embedding = False
        self.openai_client = OpenAI(api_key="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")
        self.mistral_client = MistralClient(api_key="gWH4RZAw90AiWJHMfvJvjjvh1rf7bF0M")

        if self.embedding in ['multi-qa-mpnet-base-dot-v1', 'average_word_embeddings_glove.6B.300d',
                              'average_word_embeddings_komninos', 'msmarco-bert-base-dot-v5',
                              'average_word_embeddings_levy_dependency', 'average_word_embeddings_glove.840B.300d',
                              'jinaai/jina-embeddings-v2-base-en']:

            self.sentence_embedding = True  # sentence embeddings: https://www.sbert.net/docs/pretrained_models.html
            self.sentence_embedding_model = SentenceTransformer(self.embedding)

        elif self.embedding == '':
            self.sentence_embedding = True
            self.sentence_embedding_model = SentenceTransformer(
                "jinaai/jina-embeddings-v2-base-en",  # switch to en/zh for English or Chinese
                trust_remote_code=True
            )

        elif self.embedding == 'BERT':
            with torch.no_grad():
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertModel.from_pretrained('bert-base-uncased')
                self.pretrained_embeddings = {token: self.model.get_input_embeddings()(torch.tensor(id)) for token, id
                                              in tokenizer.get_vocab().items()}
        elif self.embedding == 'word2vec':
            if os.path.isfile('experiments/word2vec'):
                self.pretrained_embeddings = KeyedVectors.load('word2vec', mmap='r')
            else:
                self.pretrained_embeddings = gensim.downloader.load('word2vec-google-news-300')  # 1662.8 MB
                self.pretrained_embeddings.init_sims(replace=True)
                self.pretrained_embeddings.save('word2vec')
        elif self.embedding == 'fasttext':
            fasttext.util.download_model('en', if_exists='ignore')  # English
            self.pretrained_embeddings = fasttext.load_model('cc.en.300.bin')
        elif self.embedding in ['text-embedding-3-large', 'text-embedding-3-small', 'mistral-embed']:
            pass  # no need to download embeddings because of API access
        else:  # default
            if os.path.isfile('experiments/word2vec'):
                self.pretrained_embeddings = KeyedVectors.load('word2vec', mmap='r')
            else:
                self.pretrained_embeddings = gensim.downloader.load('word2vec-google-news-300')  # 1662.8 MB
                self.pretrained_embeddings.init_sims(replace=True)
                self.pretrained_embeddings.save('word2vec')

    def get_openai_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input=text, model=model).data[0].embedding

    def get_mistral_embedding(self, text, model="mistral-embed"):
        text = text.replace("\n", " ")
        result = self.mistral_client.embeddings(
            model=model,
            input=text,
        )
        return result

    def word_embedding(self, w):
        if self.embedding == 'fasttext':
            return self.pretrained_embeddings.get_word_vector(w).tolist()
        elif self.embedding in ['text-embedding-3-large', 'text-embedding-3-small']:
            return np.array(self.get_openai_embedding(w, self.embedding))
        elif self.embedding == 'mistral-embed':
            return np.array(self.get_mistral_embedding(w, self.embedding).data[0].embedding)
        elif self.sentence_embedding:
            return self.sentence_embedding_model.encode(w)
        else:
            try:
                emb = self.pretrained_embeddings[w]
                return emb
            except KeyError:
                return None

    def vectorize(self, ws):
        word_list = word_tokenize(ws)

        if (self.embedding in ['text-embedding-3-large', 'text-embedding-3-small', 'mistral-embed']
                or self.sentence_embedding):

            ws = ' '.join(word_list)
            return self.word_embedding(ws)

        else:  # no contextual embeddings

            # word_list = ws.split()
            m = len(word_list)

            if m > 1:
                vs = [self.word_embedding(w) for w in word_list]
                vs = list(filter(lambda x: x is not None, vs))
                if self.embedding == 'fasttext':
                    v = np.sum(vs, axis=0) / m  # un-weighted averaging
                else:
                    v = sum(vs) / m  # un-weighted averaging
                if len(vs) == 0:
                    return self.word_embedding('none')
            else:
                v = self.word_embedding(ws)
                if v is None:
                    v = self.word_embedding('none')
        return v

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
