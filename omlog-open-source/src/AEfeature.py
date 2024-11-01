import logging
import statistics
from collections import Counter

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extractFeatures(session_train, session_test=None, unsupervised=False):
    '''
    Extract categorical and continuous features from training and testing sessions.
    The number of various categorical features in training data are returned.

    @param session_train: training session dict
    @param session_test: testing session dict
    @param unsupervised: whether in unsupervised mode
    '''
    uniq_components, uniq_events, uniq_levels = {}, {}, {}
    mean, stddev ,session_train = _extractFeatures(session_train, not unsupervised, uniq_components, uniq_events, uniq_levels)
    # print("uniq_components",uniq_components)
    # print("uniq_events",uniq_events)
    # print("uniq_levels",uniq_levels)
    num_components, num_events, num_levels = len(uniq_components), len(uniq_events), len(uniq_levels)
    logger.info(
        f'{num_components} components, {num_events} events and {num_levels} levels are extracted in training data.')

    # reserve an event id for padding
    uniq_events.setdefault('<PADDING>', num_events)

    if session_test is not None:
        # In unsupervised mode, the mean and stddev of elapsed time are computed from testing data itself
        if unsupervised:
            mean, stddev = None, None

        mean_test, stddev_test, session_test = _extractFeatures(session_test, False, uniq_components, uniq_events, uniq_levels, mean, stddev)
        num_components, num_events, num_levels = len(uniq_components), len(uniq_events), len(uniq_levels)
        logger.info(
            f'{len(uniq_components)} components, {len(uniq_events)} events and {len(uniq_levels)} levels are extracted in whole data.')
    return num_components, num_events, num_levels, uniq_events, session_train, session_test


def _extractFeatures(session_dict,
                     filter_abnormal,
                     uniq_components,
                     uniq_events,
                     uniq_levels,
                     mean=None,
                     stddev=None):
    '''
    The specific implementation of extractFeatures. Categorical features such as
    component and verbosity level are represented by its index. Continuous features
    such as time elapsed are normalized to zero mean and unit variance.
    '''
    elapsed_time_combined = []
    print("filter_abnormal:", filter_abnormal)
    if filter_abnormal:
        logger.info('filter_abnormal enabled when extracting features.')

    def extractIds(uniq_elements, entries):
        # print("entries:", entries)
        # print("change:",Counter(list(map(lambda entry: uniq_elements.setdefault(entry, len(uniq_elements)), entries))))
        return list(map(lambda entry: uniq_elements.setdefault(entry, len(uniq_elements)), entries))

    # print("session_df:",session_dict.copy().items())
    # print("filter_abnormal:",filter_abnormal)
    for key, session_df in session_dict.copy().items():
        labels = session_df['Anomaly'].tolist()
        if filter_abnormal and True in labels:
            session_dict.pop(key)
            continue

        timestamps = session_df['Timestamp'].tolist()
        elapsed_time = [0]
        elapsed_time.extend([timestamps[ind] - timestamps[ind - 1] for ind in range(1, len(timestamps))])
        elapsed_time_combined.extend(elapsed_time)
        # print("elapsed_time_combined:", elapsed_time_combined)
        session_dict[key] = {'components': extractIds(uniq_components, session_df['Component']),
                             'eventids': extractIds(uniq_events, session_df['EventId']),
                             'labels': labels,
                             'levels': extractIds(uniq_levels, session_df['Level']),
                             'templates': session_df['EventTemplate'].tolist(),
                             'elapsedtime': elapsed_time}

    # print("uniq_events:", uniq_events)
    if mean is None or stddev is None:
        mean = statistics.mean(elapsed_time_combined)
        # 标准偏差
        stddev = statistics.stdev(elapsed_time_combined)
    else:
        logger.info(f'Using mean and stddev computed from previous data.')

    for key, session in session_dict.items():
        session['elapsedtime'] = [(entry - mean) / stddev for entry in session['elapsedtime']]
    # print("mean:",mean)
    # print("stddev:",stddev)
    return mean, stddev , session_dict