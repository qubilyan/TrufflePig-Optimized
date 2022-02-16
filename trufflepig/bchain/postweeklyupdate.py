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
                tag_count_dict[tag]