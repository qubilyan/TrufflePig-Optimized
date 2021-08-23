import time
import logging

from steem.account import Account
from trufflepig.utils import error_retry

logger = logging.getLogger(__name__)


WAIT = 24.42

THRESHOLD = 'threshold'


class Poster(object):
    """A class to allow for posting and taking care of posting intervals"""
 