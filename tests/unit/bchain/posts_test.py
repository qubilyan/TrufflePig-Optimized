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

    title, post = tbpo.t