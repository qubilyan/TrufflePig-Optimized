
import logging
import os
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, \
    RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
from gensim.matutils import corpus2dense
import gensim.models.doc2vec as d2v

from trufflepig.utils import progressbar
from trufflepig.preprocessing import apply_parallel
import trufflepig.filters.stylemeasures as tfsm


logger = logging.getLogger(__name__)


# List of default features used besides topics
FEATURES = ['body_length',
            'num_sentences',
            'num_paragraphs',
            'num_words',
            'num_headings',
            'unique_words',
            'unique_ratio',
            'num_spelling_errors',
            'chars_per_word',
            'words_per_paragraph',
            'errors_per_word',
            'average_sentence_length',
            'sentence_length_variance',
            'sentence_length_skew',
            'sentence_length_kurtosis',
            'average_punctuation',
            'connectors_per_sentence',
            'pronouns_per_sentence',
            'complex_word_ratio',
            'gunning_fog_index',
            'flesch_kincaid_index',
            'smog_index',
            'automated_readability_index',
            'coleman_liau_index',
            'average_syllables',
            'syllable_variance',
            'syllable_skew',
            'syllable_kurtosis',
            'adverbs_per_sentence']

# output variables for regressor
TARGETS = ['reward', 'votes']

# Template for storing the trained model
FILENAME_TEMPLATE = 'truffle_pipeline__{time}.gz'


# tag factor punish list
PUNISH_LIST=['steem',
             'steemit',
             'crypto-news',
             'bitcoin',
             'blockchain',
             'cryptocurrency',
             'crypto',
             'dtube']


class Doc2VecModel(BaseEstimator, RegressorMixin):
    """A Doc2Vec Model following the scikit pipeline API

    NOT used in production!

    """
    def __init__(self, alpha=0.25, min_alpha=0.01, size=32,
                 window=8, min_count=5, workers=4, sample=1e-4,
                 negative=5, epochs=5, infer_steps=10):
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sample = sample
        self.negative = negative
        self.epochs=epochs
        self.model = None
        self.infer_steps = infer_steps


    def create_tagged_documents(self, document_frame):
        permalinks = document_frame.permalink
        authors = document_frame.author
        tokens = document_frame.tokens
        return [d2v.TaggedDocument(words=tks, tags=[author+'/'+permalink])
                for tks, author, permalink in zip(tokens, authors, permalinks)]

    def train(self, document_frame):
        tagged_docs = self.create_tagged_documents(document_frame)
        model = d2v.Doc2Vec(alpha=self.alpha, min_alpha=self.min_alpha,
                            size=self.size, window=self.window,
                            min_count=self.min_count, workers=self.workers,
                            sample=self.sample, negative=self.negative)
        logger.info('Building vocab')
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count,
                    epochs = self.epochs)
        logger.info('Training successfull deleting temporary data..')
        model.delete_temporary_training_data()
        self.model = model
        logger.info('...done, have fun with your new network!')

    def fit(self, document_frame, y=None):
        self.train(document_frame)
        return self

    def transform(self, document_frame):
        dim = self.model.vector_size
        inputs = np.zeros((len(document_frame), dim))
        logger.info('Transforming documents into matrix of '
                    'shape {}'.format(inputs.shape))
        tagged_docs = self.create_tagged_documents(document_frame)
        for kdx, (author, permalink) in enumerate(zip(document_frame.author,
                                                      document_frame.permalink)):
            try:
                inputs[kdx, :] = self.model.docvecs[author+'/'+permalink]
            except KeyError:
                # infer the test vector
                inputs[kdx, :] = self.model.infer_vector(tagged_docs[kdx].words,
                                                         steps=self.infer_steps)
            progressbar(kdx, len(inputs), logger=logger)
        return inputs


class KNNDoc2Vec(Doc2VecModel):
    """A K-Nearest Neighbot Doc2Vec Regressor

    Not used in production!

    """
    def __init__(self, knn=5, alpha=0.25, min_alpha=0.01, size=32,
                 window=8, min_count=5, workers=4, sample=1e-4,
                 negative=5, epochs=5, infer_steps=10):
        super().__init__(alpha=alpha, min_alpha=min_alpha, size=size,
                         window=window, min_count=min_count, workers=workers,
                         sample=sample, negative=negative, epochs=epochs,
                         infer_steps=infer_steps)
        self.knn = knn
        self.trainY = None

    def fit(self, document_frame, target_frame, sample_weight=None):
        self.trainY = target_frame.copy()
        self.trainY['doctag'] = document_frame.author + '/' + document_frame.permalink
        self.trainY = self.trainY.set_index('doctag')
        return super().fit(document_frame, target_frame)

    def predict(self, document_frame):
        logger.info('Predicting {} values'.format(len(document_frame)))
        values = self.transform(document_frame)
        results = np.zeros((len(values), self.trainY.shape[1]))
        logger.info('Finding {} nearest neighbors'.format(self.knn))
        for idx in range(len(values)):
            vector = values[idx, :]
            returns = self.model.docvecs.most_similar(positive=[vector], topn=self.knn)
            indices = [doctag for doctag, sim in returns]
            mean_vals = self.trainY.loc[indices, :].mean()
            results[idx, :] = mean_vals
            progressbar(idx, len(values), logger=logger)
        return results


class TopicModel(BaseEstimator):
    """ Gensim Latent Semantic Indexing wrapper for scikit API

    Parameters
    ----------
    no_below: int
        Filters according to minimum number of times a token must appear
    no_above: float
        Filters that a token should occur in less than `no_above` documents
    num_topics: int
        Dimensionality of topic space
    prune_at: int
        Maximum number of elements in dictionary during creation
    keep_n: int
        Maximum number of elements kept after filtering

    """
    def __init__(self, no_below, no_above, num_topics,
                 prune_at=2000000, keep_n=200000):
        self.num_topics = num_topics
        self.no_below = no_below
        self.no_above = no_above
        self.prune_at = prune_at
        self.keep_n = keep_n
        self.lsi = None
        self.tfidf = None
        self.dictionary = None

    def to_corpus(self, tokens):
        """ Transfers a list of tokens into the Gensim corpus representation

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        Returns
        -------
        list of Bag of Words representations

        """
        return [self.dictionary.doc2bow(text) for text in tokens]

    def fill_dictionary(self, tokens):
        """ Fills a dictionary

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        """
        self.dictionary = corpora.Dictionary(tokens, prune_at=self.prune_at)
        self.dictionary.filter_extremes(self.no_below, self.no_above,
                                        keep_n=self.keep_n)

    def train(self, tokens):
        """ Trains the LSI model

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        """
        self.fill_dictionary(tokens)
        corpus = self.to_corpus(tokens)
        self.tfidf = TfidfModel(corpus)
        corpus = self.tfidf[corpus]
        self.lsi = LsiModel(corpus, num_topics=self.num_topics)

    def project(self, tokens):
        """ Projects `tokens` into the N-dimensional LSI space

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        Returns
        -------
        lsi projection

        """
        corpus = self.to_corpus(tokens)
        corpus = self.tfidf[corpus]
        return self.lsi[corpus]

    def project_dense(self, tokens):
        """ Same as `project` but returns projection as numpy array """
        projection = self.project(tokens)
        result = corpus2dense(projection, self.num_topics).T
        return result

    def fit(self, data, y=None):
        """Train in scikit API language"""
        self.train(data.tokens)
        return self

    def transform(self, data):
        """Project in scikit API language"""
        return self.project_dense(data.tokens)

    def print_topics(self, n_best=10, n_words=7, topics_step=1):
        """ Returns a string of the best topics

        Parameters
        ----------
        n_best: int
            Number of topics to return
        n_words: int
            Number of words to show per topic
        topics_step: int
            Steps of printing

        Returns
        -------

        """
        if n_best is None:
            n_best = self.num_topics

        result = ''
        for topic in range(0, n_best, topics_step):
            best_words = self.lsi.show_topic(topic, n_words)
            inwords = [(self.dictionary[int(x[0])], x[1]) for x in best_words]
            wordstring = ', '.join('{}: {:0.2f}'.format(*x) for x in inwords)
            result += 'Topic {}: {}\n'.format(topic, wordstring)
        return result


def create_ngrams(tokens, n):
    """Converts tokens to ngrams with white space separator

    Returns generator for n > 1
    """
    if n == 1:
        return tokens
    return (' '.join(tokens[irun:irun + n]) for irun in range(len(tokens) - n + 1))


def create_skip_bigrams(tokens, s):
    """Creates skip grams, not used in production"""
    if s == -1:
        return tokens
    end = s + 2
    return (' '.join(tokens[irun:irun + end:s+1]) for irun in range(len(tokens) - end + 1))


class NGramTopicModel(TopicModel):
    """ Gensim Latent Semantic Indexing wrapper for scikit API

    Augments the standard model by increasing the token list by
    ngram tokens.

    Parameters
    ----------
    no_below: int
        Filters according to minimum number of times a token must appear
    no_above: float
        Filters that a token should occur in less than `no_above` documents
    num_topics: int
        Dimensionality of topic space
    prune_at: int
        Maximum dict size during construction
    keep_n: int
        Maximum dict size after filtering
    ngrams : tuple of int
        The ngrams to use e.g. (1,2) for normal tokens and bigrams

    """
    def __init__(self, no_below, no_above, num_topics, prune_at=5000000,
                 keep_n=250000, ngrams=(1,2)):
        super().__init__(no_below=no_below,
                         no_above=no_above,
                         num_topics=num_topics,
                         keep_n=keep_n,
                         prune_at=prune_at)
        if isinstance(ngrams, int):
            ngrams = (ngrams,)
        self.ngrams = ngrams

    def add_ngrams(self, tokens):
        """ Appends ngram tokens to the original ones

        Parameters
        ----------
        tokens list of str

        Returns
        -------
        list of generators over ngrams

        """