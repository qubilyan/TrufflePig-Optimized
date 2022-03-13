import logging
import time

import pandas as pd
from steem.post import Post, PostDoesNotExist
from steem.converter import Converter

import trufflepig.model as tpmo
import trufflepig.bchain.posts as tpbp
import trufflepig.bchain.getdata as tppd
import trufflepig.bchain.getaccountdata as tpaa
from trufflepig.utils import error_retry
from trufflepig.bchain.poster import Poster

logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'weekly-truffle-updates-{date}'

SPELLING_CATEGORY = ['num_spelling_errors', 'errors_per_word']

STYLE_CATEGORY = [x for x in tpmo.FEATURES if x not in SPELLING_CATEGORY]

TAGS = ['steemit', 'steemstem', 'minnowsupport', 'technology', 'utopian-io']


def compute_weekly_statistics(post_frame, pipeline, N=10, topics_step=4):
    logger.info('Computing statistics...')
    total_reward = post_frame.reward.sum()
    total_posts = len(post_frame)
    total_votes = post_frame.votes.sum()
    start_datetime = post_frame.created.min()
    end_datetime = post_frame.created.max()
    mean_reward = post_frame.reward.mean()
    median_reward = post_frame.reward.median()
    dollar_percent = (post_frame.reward < 1).sum() / len(post_frame) * 100

    # get top tags
    logger.info('Computing top tags...')
    tag_count_dict = {}
    tag_payout = {}
    for tags, reward in zip(post_frame.tags, post_frame.reward):
        for tag in tags:
            if tag not in tag_count_dict:
                tag_count_dict[tag] = 0
                tag_payout[tag] = 0
            tag_count_dict[tag] += 1
            tag_payout[tag] += reward
    counts = pd.Series(tag_count_dict, name='count')
    rewards = pd.Series(tag_payout, name='reward')
    top_tags = counts.to_frame().join(rewards).sort_values('count',
                                                          ascending=False)
    top_tags_earnings = top_tags.copy()
    top_tags = top_tags.iloc[:N, :]

    logger.info('Computing top tags earnings...')
    top_tags_earnings['per_post'] = top_tags_earnings.reward / top_tags_earnings['count']
    min_count = 500
    top_tags_earnings = top_tags_earnings[top_tags_earnings['count']
                                          >= min_count].sort_values('per_post', ascending=False)
    top_tags_earnings = top_tags_earnings.iloc[:N, :]

    logger.info('Computing bid bot stats...')
    num_articles = (post_frame.bought_votes > 0).sum()
    bid_bots_percent = num_articles / len(post_frame) * 100
    bid_bots_steem = post_frame.steem_bought_reward.sum()
    bid_bots_sbd = post_frame.sbd_bought_reward.sum()

    # get top tokens
    logger.info('Computing top words...')
    token_count_dict = {}
    for tokens in post_frame.tokens:
        for token in tokens:
            if token not in token_count_dict:
                token_count_dict[token] = 0
            token_count_dict[token] += 1
    top_words = pd.Series(token_count_dict, name='count')
    top_words = top_words.sort_values(ascending=False).iloc[:N]

    logger.info('Computing top tfidf...')
    topic_model = pipeline.named_steps['feature_generation'].transformer_list[1][1]
    tfidf = topic_model.tfidf
    dictionary = topic_model.dictionary
    sample_size = 2000
    if sample_size > len(post_frame):
        sample_frame = post_frame
    else:
        sample_frame = post_frame.sample(n=sample_size)
    corpus_tfidf = tfidf[topic_model.to_corpus(sample_frame.tokens)]
    top_tfidf = {}
    for doc in corpus_tfidf:
        for iWord, tf_idf in doc:
            iWor