
import logging
import time

from steem.post import Post

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd
import trufflepig.filters.textfilters as tftf
from trufflepig.utils import error_retry
import trufflepig.preprocessing as tppp
from trufflepig.bchain.poster import Poster


logger = logging.getLogger(__name__)


I_WAS_HERE = 'Huh? Seems like I already voted on this post, thanks for calling anyway!'

YOU_DID_NOT_MAKE_IT = """I am sorry, I cannot evaluate your post. This can have several reasons, for example, it may not be long enough, it's not in English, or has been filtered, etc."""


def post_on_call(post_frame, topN_link,
                 poster,
                 overview_permalink,
                 filter_voters=tppp.FILTER_VOTERS):
    """ Replies to users calling @trufflepig

    Parameters
    ----------
    post_frame: DataFrame
    poster: Poster
    topN_link: str