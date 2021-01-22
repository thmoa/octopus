import os
import sys

from lib.geometry import sparse_to_tensor, sparse_dense_matmul_batch_tile

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl

body_25_reg = None
face_reg = None


def joints_body25(v):
    global body_25_reg

    if body_25_reg is None:

        body_25_reg = sparse_to_tensor(
            pkl.load(open(os.path.join(os.path.dirname(__file__), '../assets/J_regressor.pkl'), 'rb'),encoding='iso-8859-1').T
        )

    return sparse_dense_matmul_batch_tile(body_25_reg, v)


def face_landmarks(v):
    global face_reg

    if face_reg is None:
        face_reg = sparse_to_tensor(
            pkl.load(open(os.path.join(os.path.dirname(__file__), '../assets/face_regressor.pkl'), 'rb'),encoding='iso-8859-1').T
        )

    return sparse_dense_matmul_batch_tile(face_reg, v)
