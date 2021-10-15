# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(model_name):
    model = Baseline(750, 1, '', 512, 0., model_name, 'self', False, 'trp_cls', '')
    return model
