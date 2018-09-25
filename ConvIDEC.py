"""
Tensorflow implementation for ConvIDEC (also known as DCEC) and ConvIDEC-DA algorithms:
    - Xifeng Guo, Xinwang Liu, En Zhu, and Jianping Yin.Deep Clustering with Convolutional Autoencoders. ICONIP 2017.
    - Xifeng Guo, En Zhu, Xinwang Liu, and Jianping Yin. Deep Embedded Clustering with Data Augmentation. ACML 2018.

Author:
    Xifeng Guo. 2018.6.30
"""

from tensorflow.keras.models import Model
from ConvDEC import ConvDEC


class ConvIDEC(ConvDEC):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10):

        super(ConvIDEC, self).__init__(input_shape, filters, n_clusters)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.model.output, self.autoencoder.output])

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)[0]
        return q

    def compile(self, optimizer='sgd', loss=['kld', 'mse'], loss_weights=[0.1, 1.0]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch(self, x, y, sample_weight=None):
        return self.model.train_on_batch(x, [y, x], sample_weight)[0]
