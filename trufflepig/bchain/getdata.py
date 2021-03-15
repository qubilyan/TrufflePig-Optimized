import logging
import os
import multiprocessing as mp
from collections import OrderedDict

import pandas as pd
from steem import Steem
from steem.blockchain import Blockchain
from steem.post import Post, PostDoesNotExist
import json
from json import JSONDecodeError

from trufflepig.utils import progressbar, error_retry, none_error_retry
import trufflepig.persist as tppe


logger = logging.getLogger(__name__)


MIN_CHARACTERS = 500

FILENAME_TEMPLATE = 'steemit_posts__{tim