import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import initializers, activations
from util import sparse_dot_adj_batch


class GraphConvolution(Layer):

    def __init__(self, output_dim, support, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        self.output_dim = output_dim
        self.support = support
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernels = []
        self.bias = None

        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        for i in range(len(self.support)):
            self.kernels.append(self.add_weight(name='kernel',
                                                shape=(input_shape[2], self.output_dim),
                                                initializer=self.kernel_initializer,
                                                trainable=True))

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        trainable=True)

        super(GraphConvolution, self).build(input_shape)

    def call(self, x):
        supports = list()
        for s, k in zip(self.support, self.kernels):
            pre_sup = K.dot(x, k)
            supports.append(sparse_dot_adj_batch(s, pre_sup))

        output = supports[0]
        for i in range(1, len(supports)):
            output += supports[i]

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim
