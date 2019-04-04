import dirt
import numpy as np
import tensorflow as tf

from dirt import matrices
from tensorflow.python.framework import ops


def perspective_projection(f, c, w, h, near=0.1, far=10., name=None):
    """Constructs a perspective projection matrix.
    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.

    Returns:
        a 4x4 `Tensor` containing the projection matrix
    """

    with ops.name_scope(name, 'PerspectiveProjection', [f, c, w, h, near, far]) as scope:
        f = 0.5 * (f[0] + f[1])
        pixel_center_offset = 0.5
        right = (w - (c[0] + pixel_center_offset)) * (near / f)
        left = -(c[0] + pixel_center_offset) * (near / f)
        top = (c[1] + pixel_center_offset) * (near / f)
        bottom = -(h - c[1] + pixel_center_offset) * (near / f)

        elements = [
            [2. * near / (right - left), 0., (right + left) / (right - left), 0.],
            [0., 2. * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]

        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))


def render_colored_batch(m_v, m_f, m_vc, width, height, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None):
    with ops.name_scope(name, "render_batch", [m_v]) as name:
        assert (num_channels == m_vc.shape[-1] == bgcolor.shape[0])

        projection_matrix = perspective_projection(camera_f, camera_c, width, height, .1, 10)

        view_matrix = matrices.compose(
            matrices.rodrigues(camera_rt.astype(np.float32)),
            matrices.translation(camera_t.astype(np.float32)),
        )

        bg = tf.tile(bgcolor.astype(np.float32)[np.newaxis, np.newaxis, np.newaxis, :],
                     (tf.shape(m_v)[0], height, width, 1))
        m_vc = tf.tile(tf.cast(m_vc, tf.float32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        m_v = tf.cast(m_v, tf.float32)
        m_v = tf.concat([m_v, tf.ones_like(m_v[:, :, -1:])], axis=2)
        m_v = tf.matmul(m_v, tf.tile(view_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))
        m_v = tf.matmul(m_v, tf.tile(projection_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))

        m_f = tf.tile(tf.cast(m_f, tf.int32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        return dirt.rasterise_batch(bg, m_v, m_vc, m_f, name=name)