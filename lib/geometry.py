import numpy as np
import tensorflow as tf


def sparse_to_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse_reorder(tf.SparseTensor(indices, coo.data, coo.shape))


def sparse_dense_matmul_batch(a, b):
    num_b = tf.shape(b)[0]
    shape = a.dense_shape

    indices = tf.reshape(a.indices, (num_b, -1, 3))
    values = tf.reshape(a.values, (num_b, -1))

    def matmul((i, bb)):
        sp = tf.SparseTensor(indices[i, :, 1:], values[i], shape[1:])
        return i, tf.sparse_tensor_dense_matmul(sp, bb)

    _, p = tf.map_fn(matmul, (tf.range(num_b), b))

    return p


def sparse_dense_matmul_batch_tile(a, b):
    return tf.map_fn(lambda x: tf.sparse_tensor_dense_matmul(a, x), b)


def batch_laplacian(v, f):
    # v: B x N x 3
    # f: M x 3

    num_b = tf.shape(v)[0]
    num_v = tf.shape(v)[1]
    num_f = tf.shape(f)[0]

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    a = tf.gather(v, v_a, axis=1)
    b = tf.gather(v, v_b, axis=1)
    c = tf.gather(v, v_c, axis=1)

    ab = a - b
    bc = b - c
    ca = c - a

    cot_a = -1 * tf.reduce_sum(ab * ca, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(ab, ca) ** 2, axis=-1))
    cot_b = -1 * tf.reduce_sum(bc * ab, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(bc, ab) ** 2, axis=-1))
    cot_c = -1 * tf.reduce_sum(ca * bc, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(ca, bc) ** 2, axis=-1))

    I = tf.tile(tf.expand_dims(tf.concat((v_a, v_c, v_a, v_b, v_b, v_c), axis=0), 0), (num_b, 1))
    J = tf.tile(tf.expand_dims(tf.concat((v_c, v_a, v_b, v_a, v_c, v_b), axis=0), 0), (num_b, 1))

    W = 0.5 * tf.concat((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a), axis=1)

    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_f * 6))

    indices = tf.reshape(tf.stack((batch_dim, J, I), axis=2), (num_b, 6, -1, 3))
    W = tf.reshape(W, (num_b, 6, -1))

    l_indices = [tf.cast(tf.reshape(indices[:, i], (-1, 3)), tf.int64) for i in range(6)]
    shape = tf.cast(tf.stack((num_b, num_v, num_v)), tf.int64)
    sp_L_raw = [tf.sparse_reorder(tf.SparseTensor(l_indices[i], tf.reshape(W[:, i], (-1,)), shape)) for i in range(6)]

    L = sp_L_raw[0]
    for i in range(1, 6):
        L = tf.sparse_add(L, sp_L_raw[i])

    dia_values = tf.sparse_reduce_sum(L, axis=-1) * -1

    I = tf.tile(tf.expand_dims(tf.range(num_v), 0), (num_b, 1))
    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_v))
    indices = tf.reshape(tf.stack((batch_dim, I, I), axis=2), (-1, 3))

    dia = tf.sparse_reorder(tf.SparseTensor(tf.cast(indices, tf.int64), tf.reshape(dia_values, (-1,)), shape))

    return tf.sparse_add(L, dia)


def compute_laplacian_diff(v0, v1, f):
    L0 = batch_laplacian(v0, f)
    L1 = batch_laplacian(v1, f)

    return sparse_dense_matmul_batch(L0, v0) - sparse_dense_matmul_batch(L1, v1)
