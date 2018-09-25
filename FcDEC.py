"""
Tensorflow implementation for FcDEC and FcDEC-DA algorithms:
    - Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.
    - Xifeng Guo, En Zhu, Xinwang Liu, and Jianping Yin. Deep Embedded Clustering with Data Augmentation. ACML 2018.

Author:
    Xifeng Guo. 2018.6.30
"""

from time import time
import numpy as np
import platform
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
import metrics


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FcDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0):

        super(FcDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.autoencoder, self.encoder = autoencoder(self.dims)

        # prepare FcDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256,
                 save_dir='results/temp', verbose=1, aug_pretrain=False):
        print('Begin pretraining: ', '-' * 60)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None and verbose > 0:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs / 10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(index=int(len(self.model.layers) / 2)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        if not aug_pretrain:
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        else:
            print('-=*'*20)
            print('Using augmentation for ae')
            print('-=*'*20)
            def gen(x, batch_size):
                if len(x.shape) > 2:  # image
                    gen0 = self.datagen.flow(x, shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        yield [batch_x, batch_x]
                else:
                    width = int(np.sqrt(x.shape[-1]))
                    if width * width == x.shape[-1]:  # gray
                        im_shape = [-1, width, width, 1]
                    else:  # RGB
                        width = int(np.sqrt(x.shape[-1] / 3.0))
                        im_shape = [-1, width, width, 3]
                    gen0 = self.datagen.flow(np.reshape(x, im_shape), shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
                        yield [batch_x, batch_x]
            self.autoencoder.fit_generator(gen(x, batch_size), steps_per_epoch=int(x.shape[0]/batch_size),
                                           epochs=epochs, callbacks=cb, verbose=verbose,
                                           workers=8, use_multiprocessing=True if platform.system() != "Windows" else False)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def load_weights(self, weights):  # load weights of FcDEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q

    def predict_labels(self, x):  # predict cluster labels using the output of clustering layer
        return np.argmax(self.predict(x), 1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def random_transform(self, x):
        if len(x.shape) > 2:  # image
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

        # if input a flattened vector, reshape to image before transform
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen = self.datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
        return np.reshape(gen.next(), x.shape)

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train_on_batch(self, x, y, sample_weight=None):
        return self.model.train_on_batch(x, y, sample_weight)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp', aug_cluster=False):
        print('Begin clustering:', '-' * 60)
        print('Update interval', update_interval)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        features = self.encoder.predict(x)
        y_pred = kmeans.fit_predict(features)
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.predict(x)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                avg_loss = loss / update_interval
                loss = 0.
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=avg_loss)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f; loss=%.5f' % (ite, acc, nmi, ari, avg_loss))

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/model_' + str(ite) + '.h5')

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            x_batch = self.random_transform(x[idx]) if aug_cluster else x[idx]
            loss += self.train_on_batch(x=x_batch, y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)

        return y_pred
