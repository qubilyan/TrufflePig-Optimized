
import pandas as pd

from trufflepig.testutils.pytest_fixtures import steem
import trufflepig.bchain.getaccountdata as tpac


def test_find_index_offset(steem):
    now = pd.datetime.utcnow()
    target = now - pd.Timedelta(days=42)
    offset, datetime = tpac.find_nearest_index(target, 'cheetah', steem)
    assert 0 < offset
    assert abs((target - datetime).seconds) < 3600*48


def test_shares_query(steem):
    result = tpac.get_delegates_and_shares('trufflepig', steem)

    assert 'smcaterpillar' in result


def test_payouts(steem):
    now_24 = pd.datetime.utcnow() + pd.Timedelta(days=1)
    result, result_steem = tpac.get_delegate_payouts('trufflepig', steem,
                                       now_24,
                                       1,
                                       0.5)

    assert 'smcaterpillar' in result
    assert 'trufflepig' not in result
    assert 'smcaterpillar' in result_steem
    assert 'trufflepig' not in result_steem


def test_bidbot_test(steem):
    min_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    max_datetime = min_datetime + pd.Timedelta(days=13)
    result = tpac.get_upvote_payments('brittuf', steem, min_datetime,
                                      max_datetime)
    assert result


def test_bidbot_test_max_time(steem):
    min_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    max_datetime = min_datetime + pd.Timedelta(days=13)
    result = tpac.get_upvote_payments('brittuf', steem, min_datetime,
                                      max_datetime, max_time=0.1)
    assert len(result) <= 1


def test_get_upvote_payments_for_accounts(steem):
    min_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    max_datetime = min_datetime + pd.Timedelta(days=5)
    accounts = ['trufflepig', 'smcaterpillar', 'brittuf']
    result = tpac.get_upvote_payments_for_accounts(accounts,
                                                   steem,
                                                   min_datetime=min_datetime,
                                                   max_datetime=max_datetime)
    assert result