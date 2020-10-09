import pandas as pd

from trufflepig.testutils.pytest_fixtures import steem
import trufflepig.bchain.checkops as tpcd


def test_check_all_ops_between(steem):
    start = pd.to_datetime('2018-01-17-13:38:30')
    end = pd.to_datetime('2018-01-17-13:41:10')
    comments = tpcd.check_all_ops_between(start, end, steem,
                                       account='originalworks',
                                       stop_after=25)
    assert comments


def test_check_all_ops_between_parallel(steem):
    start = pd.to_datetime('2018-01-17-13:39:00')
    end = pd.to_datetime('2018-01-17-13:41:00')
    comments = tpcd.check_all_ops_between_parallel(start, end, steem,
                                       account='originalwo