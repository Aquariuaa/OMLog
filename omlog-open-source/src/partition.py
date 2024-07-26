import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# partition_method: timestamp session_size:100, shuffle: store_true, train_ratio:0.8
def partition(log_df,
              partition_method,
              session_size,
              shuffle,
              train_ratio):
    if partition_method == 'session':
        session_train, session_test =  partitionBySession(log_df, train_ratio)
        logger.info('partitionBySession done.')
    elif shuffle:
        session_train, session_test = partitionByOrderShuffle(log_df, session_size, train_ratio)
        logger.info('partitionByOrderShuffle done.')
    else:
        # use this way
        session_train, session_test = partitionByOrder(log_df, session_size, train_ratio)
        logger.info('partitionByOrder done.')
        
    logger.info(f'Number of training and testing sessions are {len(session_train)} and {len(session_test)}.')
    return session_train, session_test

def generateSessions(log_df, session_size, session_cnt=0):
    '''
    Generate log sequences based on chronological order. A dict from
    seq_id to seq in DataFrame format is returned. The seq_id is 
    determined from # of seqs seen so far.
    
    @param log_df: parsed log in DataFrame format
    @param session_size: sequence length
    @param session_cnt: # of seqs seen so far
    '''
    ret = {}
    # test: 0 942699, 100
    for begin in range(0, len(log_df), session_size):
        end = begin + session_size
        ret[session_cnt] = log_df[begin: end]
        session_cnt += 1
        
    return ret, session_cnt
    
def partitionByOrder(log_df, session_size, train_ratio):
    '''
    Generate training and testing sessions based on chronological order.
    
    @param log_df: parsed log in DataFrame format
    @param session_size: controls the length (# of entries) in each session
    @param train_ratio: controls the ratio of training data
    '''
    train_len = int(len(log_df) * train_ratio)
    train_df = log_df[:train_len]
    test_df  = log_df[train_len:]
    # print("len(train_df):", len(train_df))
    # print("len(test_df):", len(test_df))

    session_train, num_train_sessions = generateSessions(train_df, session_size)
    # print("session_train:",session_train[0]) 100*15
    # print("num_train_sessions:",num_train_sessions) 37708
    session_test, num_sessions = generateSessions(test_df, session_size, num_train_sessions)
    # print("session_test:",session_test)
    # print("num_sessions:",num_sessions)
    return session_train, session_test

def partitionByOrderShuffle(log_df, session_size, train_ratio):
    '''
    Generate training and testing sessions based on chronological order.
    These sessions are generated in a random manner to simulate shuffling.
    '''
    session_total, num_sessions = generateSessions(log_df, session_size)
    session_train, session_test = {}, {}
    
    for key, session in session_total.items():
        if np.random.random() < train_ratio:
            session_train[key] = session
        else:
            session_test[key] = session
            
    return session_train, session_test

def partitionBySession(log_df, train_ratio):
    '''
    Generate training and testing sessions based on identifier in 'Session' 
    column. These sessions are generated in a random manner.
    
    @param log_df: parsed log in DataFrame format
    @param train_ratio: controls the ratio of training data
    '''
    # print("log_df:",log_df["ParameterList"].values[0][0])
    groups = log_df.groupby(by='Session')
    session_train, session_test = {}, {}
    
    for key, group in groups:
        if np.random.random() < train_ratio:
            session_train[key] = group
        else:
            session_test[key] = group
            
    return session_train, session_test
    