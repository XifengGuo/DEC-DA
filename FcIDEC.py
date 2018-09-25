"""
Tensorflow Implementation for Improved Deep Embedded Clustering FcIDEC and FcIDEC-DA:
    - Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure Preservation. IJCAI 2017.
    - Xifeng Guo, En Zhu, Xinwang Liu, and Jianping Yin. Deep Embedded Clustering with Data Augmentation. ACML 2018.

Author:
    Xifeng Guo. 2018.6.30
"""

from tensorflow.keras.models import Model
from FcDEC import FcDEC


class FcIDEC(FcDEC):
    def __init__(self, dims, n_clusters=10, alpha=1.0):
        super(FcIDEC, self).__init__(dims, n_clusters, alpha)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.model.output, self.autoencoder.output])

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)[0]
        return q

    def compile(self, optimizer='sgd', loss=['kld', 'mse'], loss_weights=[0.1, 1.0]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch(self, x, y, sample_weight=None):
        return self.model.train_on_batch(x, [y, x], sample_weight)[0]
