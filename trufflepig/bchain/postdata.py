import logging
import time

from steem.post import Post, PostDoesNotExist, VotingInvalidOnArchivedPost
from steembase.exceptions import RPCError
from steem.converter import Converter

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd
import trufflepig.filters.textfilters as tftf
from trufflepig.utils import