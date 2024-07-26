import argparse
import logging
import pandas as pd
import torch
import json
from src.feature import extractFeatures
from src.model import OMLog
from src.partition import partition
from src.vocab import buildVocab
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/BGL2.log_structured.csv',help='path of structured log file')
    parser.add_argument('--mmd_loss_path', type=str, default='xxx/result/mmd_loss.csv',help='path of mmd loss file')
    parser.add_argument('--dataset', type=str, default='BGL')

    # new parameters for online meta-learning in omlog
    parser.add_argument('--episode', default=5, type=int, help='episode for each meta-task')
    parser.add_argument('--meta_task_num', default=10, type=int, help='number of the meta-tasks of each batch')
    parser.add_argument('--online_batch_size', default=2048, type=int, help='number of the samples of each batch')
    parser.add_argument('--support_memory', default=512, type=int, help='number of the samples of each support set')
    parser.add_argument('--offline_retrain', default=False)
    parser.add_argument('--offline_memory', default=100, type=int, help='number of the history samples for offline re-train')
    parser.add_argument('--online_dsd', default=False)
    parser.add_argument('--dsd_thresh', default=0.01, type=float ,help='the threshold used by DSD to determine whether online detection')
    parser.add_argument('--test_point', default=20, type=float)

    # size-related arguments
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--input_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--session_size', default=100, type=int)
    parser.add_argument('--step_size', default=1, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    
    # autoencoder-related arguments
    parser.add_argument('--autoencoder_path',default='./model/omlog_autoencoder.pth', type=str, help='the path of trained normality detection model used for online learning')
    parser.add_argument('--online_level', default='session', type=str, choices=['session', 'log'], help='whether online learning is performed on a log sequence (session) or a single line of log (log)')
    parser.add_argument('--online_mode', default=True)
    parser.add_argument('--thresh', default=0.02, type=float ,help='the threshold used by autoencoder to determine whether a log sequence is normal')
    
    # other general arguments
    parser.add_argument('--embedding_method', default='context', type=str, choices=['context', 'semantics', 'combined'], help='the chosen embedding layer of omLog model')
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--model', default='omlog', type=str)
    parser.add_argument('--model_save_path', default='./model/omlog_model.pth', type=str)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--partition_method', default='timestamp', type=str, choices=['session', 'timestamp'])
    parser.add_argument('--pretrain_path', default='./data/wiki-news-300d-1M.vec', type=str, help='path of pretrained word embeddings')
    parser.add_argument('--shuffle', action='store_true', help='shuffle before partitioning training and testing dataset, only valid when partition_method is set to timestamp')
    parser.add_argument('--min_topk', default=0, type=int, help='only display the anomaly detection result in [min_topk, topk]')
    parser.add_argument('--topk', default=32, type=int)
    parser.add_argument('--train_ratio', default=0.5, type=float)
    parser.add_argument('--unsupervised', action='store_true', help='unsupervised training of specified model')
    args = parser.parse_args()
    args.eval_batch_size = args.online_batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using {device} device.")
    parsed_log_df = pd.read_csv(args.path)
    parsed_log_df.fillna({'EventTemplate': ''}, inplace=True)
    logger.info(f'Loading structured log file from {args.path}, {len(parsed_log_df)} log messages loaded.')
    
    # Load Pretrained word embeddings
    embedding_matrix = None
    if args.model == 'omlog' and args.embedding_method != 'context':
        if args.pretrain_path is None:
            logger.error(f'Fatal error, pretrain_path must be specified when running {args.model} with {args.embedding_method} embedding.')
            exit(0)
        else:
            embedding_matrix = buildVocab(parsed_log_df, args.pretrain_path)

    session_train, session_test = partition(parsed_log_df,
                                            args.partition_method, 
                                            args.session_size,
                                            args.shuffle,
                                            args.train_ratio)

    num_components, num_events, num_levels, uniq_events = extractFeatures(session_train, session_test, args.unsupervised)
    logger.info(f'Number of training and testing sessions after feature extraction are {len(session_train)} and {len(session_test)}.')

    eventid_templates = {}
    for ind, event_id in enumerate(parsed_log_df['EventId']):
        event_id = uniq_events.get(event_id, event_id)
        try:
            event_id = int(event_id)
        except:
            continue
        if num_events <= event_id:
            continue
        eventid_templates.setdefault(event_id, parsed_log_df['EventTemplate'][ind])
    eventid_templates = {k: eventid_templates[k] for k in sorted(eventid_templates)}
    training_uniq_templates = list(eventid_templates.values())
    logger.info(f'{len(training_uniq_templates)} unique templates identified in training data.')
    
    args.embedding_matrix = embedding_matrix
    args.num_components = num_components
    args.num_events = num_events
    args.num_levels = num_levels
    args.training_tokens_id = training_uniq_templates

    if args.model == 'omlog':
        logger.info(f'Initializing UniLog model, embedding_method: {args.embedding_method}.')
        model = OMLog(args).to(device)
    else:
        logger.error(f'Fatal error, unrecognised model {args.model}.')
        exit(0)
    logger.info(f'num_classes: {num_events}, num_layers: {args.num_layers}, input_size: {args.input_size}, hidden_size: {args.hidden_size}, topk: {args.topk}, optimizer: {args.optimizer}, lr: {args.lr}, train_ratio: {args.train_ratio}, window_size: {args.window_size}.')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.setOptimizer(optimizer)
    best_result = model.fit_evaluate(session_train, session_test, args)

    print("best_result:", best_result)
    result_save_path = './result/'+args.dataset+args.model+'_train_ratio_'+str(args.train_ratio)+'_session_size_'+str(args.session_size)+'_window_size_'+str(args.window_size)+'_step_size_'+str(args.step_size)+'_online_'+str(args.online_mode)+'_best_result.json'
    with open(result_save_path, 'w') as file:
        json.dump(best_result, file)
