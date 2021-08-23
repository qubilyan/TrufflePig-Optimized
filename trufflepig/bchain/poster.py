import time
import logging

from steem.account import Account
from trufflepig.utils import error_retry

logger = logging.getLogger(__name__)


WAIT = 24.42

THRESHOLD = 'threshold'


class Poster(object):
    """A class to allow for posting and taking care of posting intervals"""
    def __init__(self, steem, account, self_vote_limit=94, waiting_time=WAIT,
                 no_posting_key_mode=False):
        self.no_posting_key_mode = no_posting_key_mode
        self.waiting_time = waiting_time
        self.last_post_time = time.time() - self.waiting_time
        self.steem = steem
        self.accoun