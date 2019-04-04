from render import render_colored_batch
import numpy as np

from keras.engine.topology import Layer


class RenderLayer(Layer):

    def __init__(self, width, height, num_channels, vc, bgcolor, f, camera_f, camera_c,
                 camera_t=np.zeros(3), camera_rt=np.zeros(3), **kwargs):
        assert(num_channels == vc.shape[-1] == bgcolor.shape[0])

        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.vc = np.array(vc).astype(np.float32)
        self.bgcolor = np.array(bgcolor).astype(np.float32)
        self.f = np.array(f).astype(np.int32)
        self.camera_f = np.array(camera_f).astype(np.float32)
        self.camera_c = np.array(camera_c).astype(np.float32)
        self.camera_t = np.array(camera_t).astype(np.float32)
        self.camera_rt = np.array(camera_rt).astype(np.float32)

        super(RenderLayer, self).__init__(**kwargs)

    def call(self, v):
        return render_colored_batch(m_v=v, m_f=self.f, m_vc=self.vc, width=self.width, height=self.height,
                                    camera_f=self.camera_f, camera_c=self.camera_c, num_channels=self.num_channels,
                                    camera_t=self.camera_t, camera_rt=self.camera_rt, bgcolor=self.bgcolor)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.height, self.width, self.num_channels
