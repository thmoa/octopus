import tensorflow as tf
from batch_smpl import SMPL
from joints import joints_body25, face_landmarks
from keras.engine.topology import Layer


class SmplTPoseLayer(Layer):

    def __init__(self, model='assets/neutral_smpl.pkl', theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, **kwargs):
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx)
        super(SmplTPoseLayer, self).__init__(**kwargs)

    def call(self, (pose, betas, trans, v_personal)):
        verts = self.smpl(pose, betas, trans, v_personal)

        return [verts, self.smpl.v_shaped_personal, self.smpl.v_shaped]

    def compute_output_shape(self, input_shape):
        shape = input_shape[0][0], 6890, 3

        return [shape, shape, shape]


class SmplBody25FaceLayer(Layer):

    def __init__(self, model='assets/neutral_smpl.pkl', theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, **kwargs):
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx)
        super(SmplBody25FaceLayer, self).__init__(**kwargs)

    def call(self, (pose, betas, trans)):
        v_personal = tf.tile(tf.zeros((1, 6890, 3)), (tf.shape(betas)[0], 1, 1))

        v = self.smpl(pose, betas, trans, v_personal)

        return tf.concat((joints_body25(v), face_landmarks(v)), axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 95, 3
