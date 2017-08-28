from __future__ import print_function

import os

import numpy as np

from datasets import NCI60
from argparser import get_parser

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers import Input, Lambda, Layer, LSTM, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MAXLEN = 120
STEP = 100
DISPLAY_SAMPLES = 5
EPSILON_STD = 1.0

ACTIVATION = 'relu'
OPTIMIZER = 'adam'
EPOCHS = 1000
LAYERS = 1
BATCH_SIZE = 128
HIDDEN_DIM = 256
LATENT_DIM = 128
DROPOUT = 0.2


def get_nn_parser(parser=None):
    parser = parser or argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-a", "--activation",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("-e", "--epochs", type=int,
                        default=EPOCHS,
                        help="number of training epochs")
    parser.add_argument('-l', '--log', dest='logfile',
                        default=None,
                        help="log file")
    parser.add_argument("-z", "--batch_size", type=int,
                        default=BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--layers", type=int,
                        default=LAYERS,
                        help="number of RNN layers to use")
    parser.add_argument("--dropout", type=float,
                        default=DROPOUT,
                        help="dropout ratio")
    parser.add_argument("--optimizer",
                        default=OPTIMIZER,
                        help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--save",
                        default='save',
                        help="prefix of output files")
    parser.add_argument("--maxlen", type=int,
                        default=MAXLEN,
                        help="DNA chunk length")
    parser.add_argument("--step", type=int,
                        default=STEP,
                        help="step size for generating DNA chunks")
    parser.add_argument("--hidden_dim", type=int,
                        default=HIDDEN_DIM,
                        help="number of neurons in hidden layer")
    parser.add_argument("--latent_dim", type=int,
                        default=LATENT_DIM,
                        help="number of neurons in bottleneck layer")
    parser.add_argument("--display_samples", type=int,
                        default=DISPLAY_SAMPLES,
                        help="number of validation samples to display")
    return parser


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def load_by_cell_data_with_smiles(cell='BR:MCF7', subsample=False):
    df_resp = NCI60.load_dose_response(subsample=subsample, fraction=True)

    df = df_resp[df_resp['CELLNAME'] == cell].reset_index()
    df = df[['NSC', 'GROWTH', 'LOG_CONCENTRATION']]
    df = df.rename(columns={'LOG_CONCENTRATION': 'LCONC'})

    df_smiles = NCI60.load_drug_smiles()
    df = df.merge(df_smiles, on='NSC')

    df = df.set_index('NSC')
    return df


def train_nn_model(df, args):
    char_count = {}
    for s in df['smiles']:
        for c in s:
            char_count[c] = char_count.get(c, 0) + 1
    charset = sorted(char_count.keys(), key=lambda x: char_count[x], reverse=True)
    charset.insert(0, ' ')
    chars = ''.join(charset)
    charlen = len(chars)

    maxlen = args.maxlen
    ctable = CharacterTable(chars, maxlen)
    n = df.shape[0]

    df = df.sample(frac=1)
    y = np.array(df['GROWTH'])
    x = np.zeros((n, maxlen, len(chars)), dtype=np.byte)
    for i, seq in enumerate(df['smiles']):
        x[i] = ctable.encode(seq[:maxlen])

    batch_size = 32
    val_split = 0.2
    train_size = int(n * (1 - val_split) / batch_size) * batch_size
    val_size = int((n - train_size) / batch_size) * batch_size
    x_train = x[:train_size]
    x_val = x[-val_size:]
    y_train = y[:train_size]
    y_val = y[-val_size:]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    model = Sequential()
    model.add(LSTM(args.hidden_dim, input_shape=(maxlen, charlen)))

    # model.add(RepeatVector(maxlen))
    # for _ in range(args.layers):
        # model.add(LSTM(args.latent_dim, return_sequences=True))

    # model.add(TimeDistributed(Dense(charlen)))
    # model.add(Activation('softmax'))
    # model.add(Dropout(args.dropout))
    # model.add(TimeDistributed(Dense(charlen, activation='softmax')))

    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='mse', optimizer=args.optimizer)

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_val, y_val))


def main():
    description = 'Build neural network based models to predict by-cellline drug response.'
    parser = get_parser(description)
    parser = get_nn_parser(parser)
    args = parser.parse_args()

    print('Args:', args, end='\n\n')
    print('Use percent growth for dose levels in log concentration range: [{}, {}]'.format(args.min_logconc, args.max_logconc))
    print()

    cells = NCI60.all_cells() if 'all' in args.cells else args.cells

    for cell in cells:
        print('-' * 10, 'Cell line:', cell, '-' * 10)
        # df = load_by_cell_data_with_smiles(cell, drug_features=args.drug_features, scaling=args.scaling,
        #                                    min_logconc=args.min_logconc, max_logconc=args.max_logconc,
        #                                    subsample=args.subsample, feature_subsample=args.feature_subsample)
        df = load_by_cell_data_with_smiles(cell, subsample=args.subsample)

        if not df.shape[0]:
            print('No response data found')
            continue

        train_nn_model(df, args)


if __name__ == '__main__':
    main()
