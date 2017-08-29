import argparse


CELL_FEATURES = ['expression']
DRUG_FEATURES = ['descriptors']
CV = 1
FEATURE_SUBSAMPLE = 0
LOGCONC = -4.0
MIN_LOGCONC = -5.0
MAX_LOGCONC = -4.0
SCALING = 'std'
SUBSAMPLE = None

DENSE_LAYERS = [1000, 1000, 1000]

ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LOSS = 'mse'
SAVE = 'save/combo'
EPOCHS = 10
BATCH_SIZE = 32
DROPOUT = 0
LEARNING_RATE = None
BASE_LR = None
SKIP = None

SEED = 2017



def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--cell_features", nargs='+', default=CELL_FEATURES, metavar='CELL_FEATURES',
                        choices=['expression', 'mirna', 'proteome', 'all', 'expression_5platform', 'categorical'],
                        help="use one or more cell line feature sets: 'expression', 'mirna', 'proteome', 'all'; use all for ['expression', 'mirna', 'proteome']; use 'categorical' for one-hot encoded cell lines")
    parser.add_argument("--drug_features", nargs='+', default=DRUG_FEATURES, metavar='DRUG_FEATURES',
                        choices=['descriptors', 'latent', 'all', 'categorical', 'noise'],
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or one-hot encoded drugs, or random features; 'descriptors','latent', 'all', 'categorical', 'noise'")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase output verbosity')
    parser.add_argument('-a', '--activation',
                        default=ACTIVATION,
                        help='keras activation function to use in inner layers: relu, tanh, sigmoid...')
    parser.add_argument('-e', '--epochs', type=int,
                        default=EPOCHS,
                        help='number of training epochs')
    parser.add_argument('-l', '--log', dest='logfile',
                        default=None,
                        help='log file')
    parser.add_argument('-z', '--batch_size', type=int,
                        default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('-d', '--dense_layers', nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help='number of neurons in intermediate dense layers')
    parser.add_argument('--dropout', type=float,
                        default=DROPOUT,
                        help='dropout ratio')
    parser.add_argument('--lr', dest='learning_rate', type=float,
                        default=LEARNING_RATE,
                        help='learning rate')
    parser.add_argument('--base_lr', type=float,
                        default=BASE_LR,
                        help='base learning rate')
    parser.add_argument("--loss",
                        default=LOSS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument('--optimizer',
                        default=OPTIMIZER,
                        help='keras optimizer to use: sgd, rmsprop, ...')
    parser.add_argument('--save',
                        default=SAVE,
                        help='prefix of output files')
    parser.add_argument("--residual", action='store_true',
                        help="add residual skip connections to the layers")
    parser.add_argument('--warmup_lr', action='store_true',
                        help='gradually increase learning rate on start')
    parser.add_argument('--reduce_lr', action='store_true',
                        help='reduce learning rate on plateau')
    parser.add_argument('--tb', action='store_true',
                        help='use tensorboard')
    parser.add_argument('--cp', action='store_true',
                        help='checkpoint models with best val_loss')
    parser.add_argument('--seed', type=int,
                        default=SEED,
                        help='set random seed')
    parser.add_argument("--cv", type=int, default=CV,
                        help="cross validation folds")
    parser.add_argument("--feature_subsample", type=int, default=FEATURE_SUBSAMPLE,
                        help="number of features to randomly sample from each category, 0 means using all features")
    parser.add_argument("--logconc", type=float, default=LOGCONC,
                        help="log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--min_logconc", type=float, default=MIN_LOGCONC,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc",  type=float, default=MAX_LOGCONC,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--scaling", default=SCALING, metavar='SCALING',
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")
    parser.add_argument("--subsample", default=SUBSAMPLE, metavar='SUBSAMPLE',
                        choices=['naive_balancing', 'none'],
                        help="dose response subsample strategy; 'none' or 'naive_balancing'")
    parser.add_argument("--use_landmark_genes", action="store_true",
                        help="use the 978 landmark genes from LINCS (L1000) as expression features")
    parser.add_argument("--gen", action="store_true",
                        help="use generator for training and validation data")
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle training data every epoch if not using generator")
    parser.add_argument("--use_combo_score", action="store_true",
                        help="use combination score in place of percent growth in response data")

    return parser
