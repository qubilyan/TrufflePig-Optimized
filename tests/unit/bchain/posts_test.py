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
    end_datetime = start_datetime + pd.Timedelta(days=4)

    steem_per_mvests = 490
    total_posts = 70000
    total_votes = 99897788
    total_reward = 79898973

    bid_bots_sbd = 4242
    bid_bots_steem = 12
    bid_bots_percent = 99.9998

    median_reward = 0.012
    mean_reward = 6.2987347329
    dollar_percent = 69.80921393
    spelling_percent = 1.435
    style_percent = 5.5
    topic_percent = 100 - style_percent - spelling_percent

    top_posts_authors = ['michael', 'mary', 'lala']
    top_posts_titles = ['How', 'What', 'Why']
    top_posts_rewards = [9.999, 6.6, 3.333]
    top_posts_permalinks = ['how', 'what', 'why']

    top_tags = ['ketchup', 'crypto']
    top_tag_counts = [10009, 4445]
    top_tag_rewards = [3213323, 413213213]

    top_words = ['a', 'the']
    top_words_counts = [6666, 2222]

    top_tags_earnings=['hi']
    top_tags_earnings_counts=[10]
    top_tags_earnings_reward=[22]
    top_tfidf=['hi']
    top_tfidf_scores=[0.8]

    delegator_list = ['henry', 'mike', 'julia']

    topics = """Topic 0: global: 0.25, report: 0.18, sales: 0.18, research: 0.15, product: 0.13, industry: 0.13, 20132018: 0.12
Topic 1: global: -0.26, sales: -0.22, report: -0.16, 20132018: -0.16, revenue: -0.13, product: -0.13, market share: -0.12
Topic 2: blockchain: -0.23, game: 0.19, data: -0.17, c