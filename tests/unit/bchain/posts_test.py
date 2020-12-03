import pandas as pd

import trufflepig.bchain.posts as tbpo
import trufflepig.filters.textfilters as tptf
import trufflepig.preprocessing as tppp
from trufflepig.testutils import random_data


def test_comment():
    post = random_data.create_n_random_posts(1)[0]

    result = tbpo.truffle_comment(reward=post['reward'],
                                  votes=post['votes'],
                                  rank=1,
                                  topN_link='www.example.com',
                                  truffle_link='www.tf.tf')

    assert result


def test_topN_post():
    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()
    df['image_urls'] = df.body.apply(lambda x: tptf.get_image_urls(x))

    title, post = tbpo.topN_post(topN_authors=df.author,
                                 topN_permalinks=df.permalink,
                                 topN_titles=df.title,
                                 topN_filtered_bodies=df.filtered_body,
                                 topN_image_urls=df.image_urls,
                                 topN_rewards=df.reward, topN_votes=df.votes,
                                 title_date=date, truffle_link='de.de.de')

    assert post
    assert title


def test_topN_comment():
    posts = random_data.create_n_random_posts(25)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    post = tbpo.topN_comment(topN_authors=df.author,
                             topN_permalinks=df.permalink,
                             topN_titles=df.title,
                             topN_votes=df.votes,
                             topN_rewards=df.reward)

    assert post


def test_post_on_call():

    comment = tbpo.on_call_comment(reward=1000000, author='Douglas Adams', votes=42000000,
                                   topN_link='www.deep.thought',
                                   truffle_link='adsadsad.de')

    assert comment


def test_weekly_update():
    current_datetime = pd.datetime.utcnow()
    start_datetime = current_datetime - pd.Timedelta(days=10)
    end_datetime = start_datetime + pd