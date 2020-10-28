import os

import pandas as pd
from pandas.testing import assert_frame_equal

import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
from trufflepig.testutils.random_data import create_n_random_posts
from trufflepig.testutils.pytest_fixtures import tem