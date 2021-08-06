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
                                 topN_votes=df.predicted_votes,
                                 title_date=current_datetime,
                                 truffle_link=truffle_link,
                                 steem_per_mvests=steem_per_mvests)

    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%m-%d'))
    logger.info('Posting top post with permalink: {}'.format(permalink))
    poster.post(body=body,
                permalink=permalink,
                title=title,
                tags=tfbp.TAGS,
                self_vote=True)

    return permalink


def comment_on_own_top_list(sorted_post_frame, poster,
                            topN_permalink, Kstart=10, Kend=25):
    """ Adds the more ranks as a comment

    Parameters
    ----------
    sorted_post_frame: DataFrame
    poster: Poster
    topN_permalink: str
    Kstart: int
    Kend: int

    """
    df = sorted_post_frame.iloc[Kstart: Kend, :]

    comment = tfbp.topN_comment(topN_authors=df.author,
                                topN_permalinks=df.permalink,
                                topN_titles=df.title,
                                topN_rewards=df.predicted_reward,
                                topN_votes=df.predicted_votes,
                                nstart=Kstart + 1)


    logger.info('Commenting on top {} post with \n '
                '{}'.format(topN_permalink, comment))
    try:
        poster.reply(body=comment,
                     parent_author=poster.account,
                     parent_permalink=topN_permalink)
    except PostDoesNotExist:
        logger.exception('No broadcast, heh?')


def vote_and_comment_on_topK(sorted_post_frame, poster,
                             topN_permalink, overview_permalink, K=25):
    """

    Parameters
    ----------
    sorted_post_frame: DataFrame
    poster: Poster,
    topN_permalink: str
    K: int
        number of truffles to comment and upvote

    """
    logger.info('Voting and commenting on {} top truffles'.format(K))
    weight = min(850.0 / K, 100)
    topN_link =