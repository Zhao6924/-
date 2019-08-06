from keras import backend as ke
from keras.engine.topology import Layer
class MutiHeadAttention(Layer):
    def __init__(self,head_num,per_dim):
        self.head_num=head_num
        self.per_dim=per_dim
        self.output_dim=head_num*per_dim
        super(MutiHeadAttention, self).__init__()
    def build(self, input_shape):
        self.WQ=self.add_weight(name='Q',shape=(input_shape[0][-1],self.output_dim),initializer='glorot_uniform',
                                  trainable=True)
        self.WK=self.add_weight(name='K',shape=(input_shape[0][-1],self.output_dim),initializer='glorot_uniform',trainable=True)

        self.WV=self.add_weight(name='V',shape=(input_shape[0][-1],self.output_dim),initializer='glorot_uniform',trainable=True)
        super(MutiHeadAttention, self).build(input_shape)
    def call(self, inputs):
        q, k, v = inputs[: 3]
        shapes=ke.shape(q)
        Q=ke.dot(q,self.WQ);
        Q=ke.reshape(Q,(-1,shapes[1],self.head_num,self.per_dim))
        Q=ke.permute_dimensions(Q,(0,2,1,3))

        K = ke.dot(k, self.WK);
        K = ke.reshape(K, (-1, shapes[1], self.head_num, self.per_dim))
        K = ke.permute_dimensions(K, (0, 2, 1, 3))

        V = ke.dot(v, self.WV);
        V = ke.reshape(V, (-1, shapes[1], self.head_num, self.per_dim))
        V = ke.permute_dimensions(V, (0, 2, 1, 3))

        QK=ke.batch_dot(Q,K,axes=[3,3])/(self.per_dim**0.50)
        SofQk=ke.softmax(QK)
        QKV=ke.batch_dot(SofQk,V,axes=[3,2])

        out=ke.permute_dimensions(QKV,(0,2,1,3))

        o = ke.reshape(out, (-1, ke.shape(out)[1], self.output_dim))
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
