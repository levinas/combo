#! /usr/bin/env python

from __future__ import division, print_function

import collections
import logging
import os
import threading

import numpy as np
import pandas as pd

from itertools import cycle, islice

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparser
from datasets import NCI60


logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    import random
    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, verbose):
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)



def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.dropout > 0:
        ext += '.DR={}'.format(args.dropout)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.no_gen:
        ext += '.ng'
    for i, n in enumerate(args.dense_layers):
        if n > 0:
            ext += '.D{}={}'.format(i+1, n)

    return ext


class ComboDataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, seed, val_split=0.2, shuffle=True,
                 cell_features=['expression'], drug_features=['descriptors'],
                 use_landmark_genes=False, feature_subsample=None,
                 scaling='std', scramble=False):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        seed: integer
            seed for random generation
        val_split : float, optional (default 0.2)
            fraction of data to use in validation
        cell_features: list of strings from 'expression', 'expression_5platform', 'mirna', 'proteome', 'all', 'categorical' (default ['expression'])
            use one or more cell line feature sets: gene expression, microRNA, proteome
            use 'all' for ['expression', 'mirna', 'proteome']
            use 'categorical' for one-hot encoded cell lines
        drug_features: list of strings from 'descriptors', 'latent', 'all', 'noise' (default ['descriptors'])
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
            trained on NSC drugs, or both; use random features if set to noise
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        use_landmark_genes: True or False
            only use LINCS1000 landmark genes
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        """

        np.random.seed(seed)

        df = NCI60.load_combo_response(fraction=True)
        logger.info('Loaded {} unique (CL, D1, D2) response sets.'.format(df.shape[0]))

        if 'all' in cell_features:
            self.cell_features = ['expression', 'mirna', 'proteome']
        else:
            self.cell_features = cell_features

        if 'all' in drug_features:
            self.drug_features = ['descriptors', 'latent']
        else:
            self.drug_features = drug_features

        for fea in self.cell_features:
            if fea == 'expression' or fea == 'expression_u133p2':
                self.df_cell_expr = NCI60.load_cell_expression_u133p2(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'expression_5platform':
                self.df_cell_expr = NCI60.load_cell_expression_5platform(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'mirna':
                self.df_cell_mirna = NCI60.load_cell_mirna(ncols=feature_subsample, scaling=scaling)
                df = df.merge(self.df_cell_mirna[['CELLNAME']], on='CELLNAME')
            elif fea == 'proteome':
                self.df_cell_prot = NCI60.load_cell_proteome(ncols=feature_subsample, scaling=scaling)
                df = df.merge(self.df_cell_prot[['CELLNAME']], on='CELLNAME')
            elif fea == 'categorical':
                df_cell_ids = df[['CELLNAME']].drop_duplicates()
                cell_ids = df_cell_ids['CELLNAME'].map(lambda x: x.replace(':', '.'))
                df_cell_cat = pd.get_dummies(cell_ids)
                df_cell_cat.index = df_cell_ids['CELLNAME']
                self.df_cell_cat = df_cell_cat.reset_index()

        for fea in self.drug_features:
            if fea == 'descriptors':
                self.df_drug_desc = NCI60.load_drug_descriptors(ncols=feature_subsample, scaling=scaling)
                df = df[df['NSC1'].isin(self.df_drug_desc['NSC']) & df['NSC2'].isin(self.df_drug_desc['NSC'])]
            elif fea == 'latent':
                self.df_drug_auen = NCI60.load_drug_autoencoded_AG(ncols=feature_subsample, scaling=scaling)
                df = df[df['NSC1'].isin(self.df_drug_auen['NSC']) & df['NSC2'].isin(self.df_drug_auen['NSC'])]
            elif fea == 'noise':
                df_drug_ids = df[['NSC1']].drop_duplicates()
                df_drug_ids.columns = ['NSC']
                noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
                df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                       columns=['RAND-{:03d}'.format(x) for x in range(500)])
                self.df_drug_rand = df_rand.reset_index()

        logger.info('Filtered down to {} rows with matching information.'.format(df.shape[0]))

        if shuffle:
            df = df.sample(frac=1.0, random_state=seed)

        self.df_response = df

        if scramble:
            growth = df[['GROWTH']]
            random_growth = growth.iloc[np.random.permutation(np.arange(growth.shape[0]))].reset_index()
            self.df_response[['GROWTH']] = random_growth['GROWTH']
            logger.warn('Randomly shuffled dose response growth values.')

        logger.info('Distribution of dose response:')
        logger.info(self.df_response[['GROWTH']].describe())

        self.total = df.shape[0]
        self.n_val = int(self.total * val_split)
        self.n_train = self.total - self.n_val
        logger.info('Rows in train: {}, val: {}'.format(self.n_train, self.n_val))

        self.cell_df_dict = {'expression': 'df_cell_expr',
                             'mirna': 'df_cell_mirna',
                             'proteome': 'df_cell_prot',
                             'categorical': 'df_cell_cat'}

        self.drug_df_dict = {'descriptors': 'df_drug_desc',
                             'latent': 'df_drug_auen',
                             'noise': 'df_drug_rand'}

        self.input_features = collections.OrderedDict()
        self.feature_shapes = {}
        for fea in self.cell_features:
            feature_type = 'cell.' + fea
            feature_name = 'cell.' + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            self.input_features[feature_name] = feature_type
            self.feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        for drug in ['drug1', 'drug2']:
            for fea in self.drug_features:
                feature_type = 'drug.' + fea
                feature_name = drug + '.' + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                self.input_features[feature_name] = feature_type
                self.feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        logger.info('Input features shapes:')
        for k, v in self.input_features.items():
            logger.info('  {}: {}'.format(k, self.feature_shapes[v]))

        self.input_dim = sum([np.prod(self.feature_shapes[x]) for x in self.input_features.values()])
        logger.info('Total input dimensions: {}'.format(self.input_dim))

    def load_data(self):
        df_train = self.df_response.iloc[:self.n_train, :]
        df_val = self.df_response.iloc[self.n_train:, :]

        y_train = df_train['GROWTH'].values
        y_val = df_val['GROWTH'].values

        x_train_list = []
        x_val_list = []

        for fea in self.cell_features:
            df_cell = getattr(self, self.cell_df_dict[fea])
            df_x_train = pd.merge(df_train[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            df_x_val = pd.merge(df_val[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            x_train_list.append(df_x_train.drop(['CELLNAME'], axis=1).values)
            x_val_list.append(df_x_val.drop(['CELLNAME'], axis=1).values)

        for drug in ['NSC1', 'NSC2']:
            for fea in self.drug_features:
                df_drug = getattr(self, self.drug_df_dict[fea])
                df_x_train = pd.merge(df_train[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                df_x_val = pd.merge(df_val[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                x_train_list.append(df_x_train.drop([drug, 'NSC'], axis=1).values)
                x_val_list.append(df_x_val.drop([drug, 'NSC'], axis=1).values)

        return x_train_list, y_train, x_val_list, y_val


class ComboDataGenerator(object):
    """Generate training, validation or testing batches from loaded data
    """
    def __init__(self, data, partition='train', batch_size=32):
        self.lock = threading.Lock()
        self.data = data
        self.partition = partition
        self.batch_size = batch_size

        if partition == 'train':
            self.cycle = cycle(range(data.n_train))
            self.num_data = data.n_train
        elif partition == 'val':
            self.cycle = cycle(range(data.total)[-data.n_val:])
            self.num_data = data.n_val
        else:
            raise Exception('Data partition "{}" not recognized.'.format(partition))

    def flow(self):
        """Keep generating data batches
        """
        while 1:
            self.lock.acquire()
            indices = list(islice(self.cycle, self.batch_size))
            self.lock.release()

            df = self.data.df_response.iloc[indices, :]
            y = df['GROWTH'].values

            x_list = []

            for fea in self.data.cell_features:
                df_cell = getattr(self.data, self.data.cell_df_dict[fea])
                df_x = pd.merge(df[['CELLNAME']], df_cell, on='CELLNAME', how='left')
                # print(df_x.head(5))
                x_list.append(df_x.drop(['CELLNAME'], axis=1).values)

            for drug in ['NSC1', 'NSC2']:
                for fea in self.data.drug_features:
                    df_drug = getattr(self.data, self.data.drug_df_dict[fea])
                    df_x = pd.merge(df[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                    # print(df_x.head(5))
                    x_list.append(df_x.drop([drug, 'NSC'], axis=1).values)

            yield x_list, y


def test_generator(loader):
    gen = ComboDataGenerator(loader).flow()
    x_list, y = next(gen)
    for x in x_list:
        print(x.shape)
    print(y.shape)


def test_loader(loader):
    x_train_list, y_train, x_val_list, y_val = loader.load_data()
    print('x_train shapes:')
    for x in x_train_list:
        print(x.shape)
    print('y_train shape:', y_train.shape)

    print('x_val shapes:')
    for x in x_val_list:
        print(x.shape)
    print('y_val shape:', y_val.shape)


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.best_val_loss = np.Inf
        self.best_model = None

    def on_batch_end(self, batch, logs={}):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = self.model
            self.best_val_loss = val_loss


def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
                        activation='relu', residual=False):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


def main():
    description = 'Build neural network based models to predict tumor response to drug pairs.'
    parser = argparser.get_parser(description)
    args = parser.parse_args()

    set_seed(args.seed)
    ext = extension_from_parameters(args)
    prefix = args.save + ext
    logfile = args.logfile if args.logfile else prefix+'.log'
    set_up_logger(logfile, args.verbose)
    logger.info(args)

    loader = ComboDataLoader(seed=args.seed, use_landmark_genes=args.use_landmark_genes)
    # test_loader(loader)
    # test_generator(loader)

    train_gen = ComboDataGenerator(loader, batch_size=args.batch_size).flow()
    val_gen = ComboDataGenerator(loader, partition='val', batch_size=args.batch_size).flow()

    train_steps = int(loader.n_train / args.batch_size)
    val_steps = int(loader.n_val / args.batch_size)

    input_models = {}
    for fea_type, shape in loader.feature_shapes.items():
        box = build_feature_model(input_shape=shape, name=fea_type, dense_layers=args.dense_layers)
        box.summary()
        input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        input_model = input_models[fea_type]
        encoded = input_model(fea_input)
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense_layers):
        x = h
        h = Dense(layer, activation=args.activation)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    model = Model(inputs, output)
    model.summary()

    if args.cp:
        model_json = model.to_json()
        with open(prefix+'.model.json', 'w') as f:
            print(model_json, file=f)

    optimizer = optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
    base_lr = args.base_lr or K.get_value(optimizer.lr)
    if args.learning_rate:
        K.set_value(optimizer.lr, args.learning_rate)

    model.compile(loss=args.loss, optimizer=optimizer, metrics=[mae, r2])

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size/100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5-epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    warmup_lr = LearningRateScheduler(warmup_scheduler)
    checkpointer = ModelCheckpoint(args.save+ext+'.weights.h5', save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))
    history_logger = LoggingCallback(logger.debug)
    history = LossHistory()

    callbacks = [history_logger, history]
    if args.reduce_lr:
        callbacks.append(reduce_lr)
    if args.warmup_lr:
        callbacks.append(warmup_lr)
    if args.cp:
        callbacks.append(checkpointer)
    if args.tb:
        callbacks.append(tensorboard)

    if args.no_gen:
        x_train_list, y_train, x_val_list, y_val = loader.load_data()
        model.fit(x_train_list, y_train,
                  batch_size=args.batch_size,
                  shuffle=args.shuffle,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  validation_data=(x_val_list, y_val))
    else:
        model.fit_generator(train_gen, train_steps,
                            epochs=args.epochs,
                            callbacks=callbacks,
                            validation_data=val_gen, validation_steps=val_steps)

    if args.cp:
        history.best_model.save(prefix+'.model.h5')

    logger.handlers = []


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
