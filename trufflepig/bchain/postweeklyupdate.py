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


def compute_weekly_statistics(post_frame, pipeline, N=10, topics_s