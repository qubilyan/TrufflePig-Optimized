import logging
import os
import multiprocessing as mp
import gc

import pandas as pd
import numpy as np
import scipy.stats as spst
from steem.amount import Amount

import trufflepig.filters.stylemeasures as tfsm
import trufflepig.filters.textfilters as tftf
import trufflepig.bchain.getaccountdata as tfga
from trufflepig.filters.blacklist import BUILD_A_WHALE_BLACKLIST


logger = logging.getLogger(__name__)


FILTER_TAGS = ('mitnebcurationtrail', 'informationwar', 'truth', 'conspiracy',
               'vaccines', 'contest', 'giveaway', 'deutsch', 'kr', 'kr-newbie',
               'nsfw', 'sex', 'daily', 'photofeed', 'gambling',
               # other weird stuff
               'steemsilvergold', 'horoscope', 'guns', 'investing', 'tib',
               # Somehow religious texts do not work in combination with others
               # maybe I need a bot just to rate spiritual content
               # for simplicity let's ignore them for now,
               # sorry, no releigious truffles in the near future!
               'bible', 'faith', 'spiritual', 'christianity', 'steemchurch',
               # Filter translations for utoptian
               'translations', 'translation')


# Stay out of the whale wars!
FILTER_AUTHORS = ('haejin', 'ew-and-patterns', 'caladium',
                  'cryptopassion', 'thirdeye7', 'shariarahammad')

# Get out plagiarismos!
FILTER_VOTERS = ('cheetah',)


def filter_duplicates(frame):
    """ Filters out duplicate entries based on author and permalink

    Filtering is inplace!

    Parameters
    ----------
    frame: DataFrame

    Returns
    -------
    DataFrame

    """
    old_len = len(frame)
    frame.drop_duplicates(subset=['author', 'permalink'],
                                     keep='last', inplace=True)
    if len(frame) < old_len:
        logger.info('Filtered {} duplicates kept {} '
                    'posts'.format(old_len - len(frame), len(frame)))
    return frame


def apply_parallel(function, iterable, ncores, chunksize=1000):
    """ Applies a `function` in parallel on `ncores`.

    Parameters
    ----------
    function: callable
    iterable: list, tuple, etc.
    ncores: int
        The number of jobs started
    chunksize: int
        Size of chunk submitted to pool

    Returns
    -------
    List of function outputs

    """
    if ncores == 1:
        return [function(x) for x in iterable]
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(ncores)

        results = [x for x in pool.imap(function, iterable, chunksize)]

        pool.close()
        pool.join()

        return results


def preprocess(post_df, ncores=4, chunksize=500,
               detect_seed=42, detect_max_length=2500,
               min_en_prob=0.9,
               min_max_body_length=(500, 35000),
               min_max_letter_ratio=(0.5, 0.85),
               min_max_num_paragraphs=(2, 250),
               min_max_num_words=(100, 12500),
               min_max_num_sentences=(5, 1250),
               min_max_words_per_paragraph=(10, 1250),
               max_errors_per_word=0.2,
               min_max_average_punctuation=(1.05, 5),
               min_max_average_sentence_length=(10, 350),
               filter_tags=FILTER_TAGS,
               filter_authors=FILTER_AUTHORS + BUILD_A_WHALE_BLACKLIST,
               filter_voters=FILTER_VOTERS,
               dropna=True):
    """ Preprocessing of raw steemit posts, filters and adds features

    All filtering happening inplace!

    Parameters
    ----------
    post_df: DataFrame
        Raw steemit posts, needs to contain
            * author
            * permalink
            * body
            * title
            * votes
            * reward
    ncores: int
        Some stuff is executed in parallel, these are the number of jobs
    chunksize: int
        Size of multiprocessing chunk
    detect_seed: int
        Seed value for language detection
    detect_max_length: int
        Maximum character size for language detection
    min_en_prob: float
        0 < min_en_prob <= 1, Minimum detection probability to classify a
        post as English
    min_max_body_length: tuple of int
        Boundaries for allowed (filtered) body length
    min_max_letter_ratio: tuple of float
        Boundaries for letters vs punctuation ratio
    min_max_num_paragraphs: tuple of int
        Boundaries for number of paragraphs
    min_max_num_words: tuple of int
        Boundaries for number of words
    min_max_num_sentences: tuple of int
        Boundaries of number of sentences
    min_max_words_per_paragraph:
        Boundaries for min max average words per paragraph
    max_errors_per_word: float
        Threshold of maximum spelling errors per word allowed
    min_max_average_punctuation: tuple of float
        Boundaries for average punctuation per sentence
    min_max_average_sentence_length: tuple of float
        Boundaries for average sentence length
    filter_tags: tuple of string
        Tags to be filtered like 'sex', 'nsfw' or controversial stuff like
        'vaccines'.
    filter_authors: tuple of string
        Authors to be filtered...
    filter_voters: tuple of string
        If vored by one of them post is excluded
    dropna: bool
        If NaN rows should be dropped

    Returns
    -------
    Filtered frame

    """
    logger.info('Filtering duplicates of {} posts'.format(len(post_df)))
    post_df = filter_duplicates(post_df)

    logger.info('Filtering authors {}'.format(filter_authors))
    filter_authors = set(filter_authors)
    author_filter = post_df.author.apply(lambda x: x in filter_authors)
    to_drop = post_df.loc[author_filter]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_df)))

    logger.info('Filtering voted by {}'.format(filter_voters))
    filter_voters = set(filter_voters)
    voted_by = post_df.active_votes.apply(lambda x: tftf.voted_by(x, filter_voters))
    to_drop = post_df.loc[voted_by]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_df)))

    logger.info('Filtering tags {}'.format(filter_tags))
    filter_tags = set(filter_tags)
    tag_filter = post_df.tags.apply(lambda x: tftf.is_in_filter_tags(x, filter_tags))
    to_drop = post_df.loc[tag_filter]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_df)))

    logger.info('Filtering images and links')
    post_df['filtered_body'] = post_df.body.apply(lambda x:
                                                  tftf.filter_images_and_links(x))

    logger.info('Filtering quotes')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x: tftf.filter_quotes(x))

    logger.info('Counting and filtering headings')
    post_df['num_headings'] = post_df.filtered_body.apply(lambda x:
                                                          tfsm.count_headings(x))
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_headings(x))

    logger.info('Filtering html')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x: tftf.
                                                           filter_html_tags(x))

    logger.info('Filtering urls')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_urls(x))

    logger.info('Filtering formatting')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_formatting(x))

    logger.info('Filtering special characters')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_special_characters(x))

    logger.info('Counting paragraphs')
    post_df['num_paragraphs'] = post_df.filtered_body.apply(lambda x:
                                                        tfsm.count_paragraphs(x))
    to_drop = post_df.loc[(post_df.num_paragraphs < min_max_num_paragraphs[0]) |
                          (post_df.num_paragraphs > min_max_num_paragraphs[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to num paragraphs limits {} '
                'kept {} posts.'.format(min_max_num_paragraphs, len(post_df)))

    logger.info('Calculating length')
    post_df['body_length'] = post_df.filtered_body.apply(lambda x: len(x))
    to_drop = post_df.loc[(post_df.body_length < min_max_body_length[0]) |
                          (post_df.body_length > min_max_body_length[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to body limits {} '
                'kept {} posts.'.format(min_max_body_length, len(post_df)))

    logger.info('Counting letters')
    post_df['letter_count'] = post_df.filtered_body.apply(lambda x: tfsm.count_letters(x))
    post_df['letter_ratio'] = post_df.letter_count / post_df.body_length
    to_drop = post_df.loc[(post_df.letter_ratio < min_max_letter_ratio[0]) |
                          (post_df.letter_ratio > min_max_letter_ratio[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to letter ratio limits {} '
                'kept {} posts.'.format(min_max_letter_ratio, len(post_df)))

    logger.info('Splitting into sentences')
    post_df['filtered_sentences'] = post_df.filtered_body.apply(lambda x:
                                                            tfsm.split_into_sentences(x))
    post_df['num_sentences'] = post_df.filtered_sentences.apply(lambda x: len(x))
    to_drop = post_df.loc[(post_df.num_sentences < min_max_num_sentences[0]) |
                          (post_df.num_sentences > min_max_num_sentences[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to num sentences lim