import numpy as np
import pandas as pd

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
from trufflepig.testutils.random_data import create_n_random_posts


def test_tag_measure():
    posts = create_n_random_posts(100)

    post_frame = pd.DataFrame(posts)

    