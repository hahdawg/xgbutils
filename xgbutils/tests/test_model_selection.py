import numpy as np
import pandas as pd

from .. import model_selection as ms


def _date_range_to_index(start_date, end_date):
    """
    Translate a date pair like ("1/3/2017", "1/7/2017") to
    np.arange(3, 8).

    Note: The test cases always use the same month.
    """
    start_index = pd.to_datetime(start_date).day
    end_index = pd.to_datetime(end_date).day + 1
    return np.arange(start_index, end_index)


def test_kfoldts():
    date_index = pd.date_range("1/1/2017", "1/31/2017", freq="D")
    fold_dates = [
                  [("1/1/2017", "1/10/2017"), ("1/11/2017", "1/31/2017")],
                  [("1/10/2017", "1/20/2017"), ("1/25/2017", "1/26/2017")],
                  [("1/1/2017", "1/1/2017"), ("1/2/2017", "1/2/2017")]
                 ]
    kft = ms.KFoldTs(date_index=date_index, fold_dates=fold_dates)
    for (train_dates, test_dates), (train_index, test_index) in zip(fold_dates, kft.split()):
        assert (_date_range_to_index(*train_dates) == train_index + 1).all()
        assert (_date_range_to_index(*test_dates) == test_index + 1).all()
