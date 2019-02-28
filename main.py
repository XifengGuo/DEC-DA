from tensorflow.keras.optimizers import SGD, Adam
import os
import numpy as np
from time import time
from FcDEC import FcDEC
from FcIDEC import FcIDEC
from ConvDEC import ConvDEC
from ConvIDEC import ConvIDEC
from datasets import load_data, load_data_conv
import metrics


def _get_data_and_model(args):
    # prepare dataset
    if args.method in ['FcDEC', 'FcIDEC', 'FcDEC-DA', 'FcIDEC-DA']:
        x, y = load_data(args.dataset)
    elif args.method in ['ConvDEC', 'ConvIDEC', 'ConvDEC-DA', 'ConvIDEC-DA']:
        x, y = load_data_conv(args.dataset)
    else:
        raise ValueError("Invalid value for method, which can only be in ['FcDEC', 'FcIDEC', 'ConvDEC', 'ConvIDEC', "
                         "'FcDEC-DA', 'FcIDEC-DA', 'ConvDEC-DA', 'ConvIDEC-DA']")

    # prepare optimizer
    if args.optimizer in ['sgd', 'SGD']:
        optimizer = SGD(args.lr, 0.9)
    else:
        optimizer = Adam()

    # prepare the model
    n_clusters = len(np.unique(y))
    if 'FcDEC' in args.method:
        model = FcDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters)
        model.compile(optimizer=optimizer, loss='kld')
    elif 'FcIDEC' in args.method:
        model = FcIDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters)
        model.compile(optimizer=optimizer, loss=['kld', 'mse'], loss_weights=[0.1, 1.0])
    elif 'ConvDEC' in args.method:
        model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=n_clusters)
        model.compile(optimizer=optimizer, loss='kld')
    elif 'ConvIDEC' in args.method:
        model = ConvIDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=n_clusters)
        model.compile(optimizer=optimizer, loss=['kld', 'mse'], loss_weights=[0.1, 1.0])
    else:
        raise ValueError("Invalid value for method, which can only be in ['FcDEC', 'FcIDEC', 'ConvDEC', 'ConvIDEC', "
                         "'FcDEC-DA', 'FcIDEC-DA', 'ConvDEC-DA', 'ConvIDEC-DA']")

    # if -DA method, we'll force aug_pretrain and aug_cluster is True
    if '-DA' in args.method:
        args.aug_pretrain = True
        args.aug_cluster = True

    return (x, y), model


def train(args):
    # get data and model
    (x, y), model = _get_data_and_model(args)
    model.model.summary()

    # pretraining
    t0 = time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):  # load pretrained weights
        model.autoencoder.load_weights(args.pretrained_weights)
    else:  # train
        pretrain_optimizer = SGD(1.0, 0.9) if args.method in ['FcDEC', 'FcIDEC', 'FcDEC-DA', 'FcIDEC-DA'] else 'adam'
        model.pretrain(x, y, optimizer=pretrain_optimizer, epochs=args.pretrain_epochs, batch_size=args.batch_size,
                       save_dir=args.save_dir, verbose=args.verbose, aug_pretrain=args.aug_pretrain)
    t1 = time()
    print("Time for pretraining: %ds" % (t1 - t0))

    # clustering
    y_pred = model.fit(x, y=y, maxiter=args.maxiter, batch_size=args.batch_size, update_interval=args.update_interval,
                       save_dir=args.save_dir, aug_cluster=args.aug_cluster)
    if y is not None:
        print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
              (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
    t2 = time()
    print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
    print('='*60)


def test(args):
    assert args.weights is not None
    (x, y), model = _get_data_and_model(args)
    model.model.summary()

    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)
    y_pred = model.predict_labels(x)
    print('acc=%.4f, nmi=%.4f, ari=%.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
    print('End testing:', '-' * 60)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--method', default='FcDEC',
                        choices=['FcDEC', 'FcIDEC', 'ConvDEC', 'ConvIDEC', 'FcDEC-DA', 'FcIDEC-DA', 'ConvDEC-DA', 'ConvIDEC-DA'],
                        help="Clustering algorithm")
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'mnist-test', 'usps', 'fmnist'],
                        help="Dataset name to train on")
    parser.add_argument('-d', '--save-dir', default='results/temp',
                        help="Dir to save the results")

    # Parameters for pretraining
    parser.add_argument('--aug-pretrain', action='store_true',
                        help="Whether to use data augmentation during pretraining phase")
    parser.add_argument('--pretrained-weights', default=None, type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs', default=500, type=int,
                        help="Number of epochs for pretraining")
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help="Verbose for pretraining")

    # Parameters for clustering
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Testing the clustering performance with provided weights")
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help="Model weights, used for testing")
    parser.add_argument('--aug-cluster', action='store_true',
                        help="Whether to use data augmentation during clustering phase")
    parser.add_argument('--optimizer', default='adam', type=str,
                        help="Optimizer for clustering phase")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="learning rate during clustering")
    parser.add_argument('--batch-size', default=256, type=int,
                        help="Batch size")
    parser.add_argument('--maxiter', default=2e4, type=int,
                        help="Maximum number of iterations")
    parser.add_argument('-i', '--update-interval', default=140, type=int,
                        help="Number of iterations to update the target distribution")
    parser.add_argument('--tol', default=0.001, type=float,
                        help="Threshold of stopping training")
    args = parser.parse_args()
    print('+' * 30, ' Parameters ', '+' * 30)
    print(args)
    print('+' * 75)

    # testing
    if args.testing:
        test(args)
    else:
        train(args)
