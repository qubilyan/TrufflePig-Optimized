import logging
import time

from steem.post import Post, PostDoesNotExist, VotingInvalidOnArchivedPost
from steembase.exceptions import RPCError
from steem.converter import Converter

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd
import trufflepig.filters.textfilters as tftf
from trufflepig.utils import error_retry
from trufflepig.bchain.poster import Poster


logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'daily-truffle-picks-{date}'

TRENDING_PERMALINK_TEMPLATE = 'non-bot-trending-{date}'


def post_topN_list(sorted_post_frame, poster,
                   current_datetime, overview_permalink, N=10):
    """ Post the toplist to the blockchain

    Parameters
    ----------
    sorted_post_frame: DataFrame
    poster: Poster
    current_datetime: datetime
    N: int
        Size of top list

    Returns
    -------
    permalink to new post

    """
    df = sorted_post_frame.iloc[:N, :]

    logger.info('Creating top {} post'.format(N))
    first_image_urls = df.body.apply(lambda x: tftf.get_image_urls(x))

    steem_per_mvests = Converter(poster.steem).steem_per_mvests()
    truffle_link = 'https://steemit.com/steemit/@{}/{}'.format(poster.account,
                                                               overview_permalink)

    title, body = tfbp.topN_post(topN_authors=df.author,
                                 topN_permalinks=df.permalink,
                                 topN_titles=df.title,
                                 topN_filtered_bodies=df.filtered_body,
                                 topN_image_urls=first_image_urls,
                                 topN_rewards=df.predicted_reward,
               