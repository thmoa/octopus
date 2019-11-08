#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl


_cache = None


def get_bodypart_vertex_ids():
    global _cache

    if _cache is None:
        with open(os.path.join(os.path.dirname(__file__), '../assets/bodyparts.pkl'), 'rb') as fp:
            _cache = pkl.load(fp,encoding='iso-8859-1')

    return _cache


def regularize_laplace():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 8.
    reg[v_ids['hand_l']] = 5.
    reg[v_ids['hand_r']] = 5.
    reg[v_ids['fingers_l']] = 8.
    reg[v_ids['fingers_r']] = 8.
    reg[v_ids['foot_l']] = 5.
    reg[v_ids['foot_r']] = 5.
    reg[v_ids['toes_l']] = 8.
    reg[v_ids['toes_r']] = 8.
    reg[v_ids['ear_l']] = 10.
    reg[v_ids['ear_r']] = 10.

    return reg


def regularize_symmetry():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 10.
    reg[v_ids['hand_l']] = 10.
    reg[v_ids['hand_r']] = 10.
    reg[v_ids['foot_l']] = 10.
    reg[v_ids['foot_r']] = 10.
    reg[v_ids['ear_l']] = 5.
    reg[v_ids['ear_r']] = 5.

    return reg
