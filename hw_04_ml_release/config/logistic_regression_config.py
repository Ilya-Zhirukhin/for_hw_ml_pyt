from easydict import EasyDict

from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria
from math import e
cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization

def choice_weights_init_type():
    print('input: uniform or he or xavier or normal')
    n = str(input('input: ' ))
    if n == 'uniform':
        cfg.weights_init_type = WeightsInitType.uniform
        cfg.weights_init_kwargs = {'epsilon': e}

    elif n == 'he':
        # training
        cfg.weights_init_type = WeightsInitType.he

        cfg.weights_init_kwargs = {'n_in': 0.2}

    elif n == 'xavier':
        # training
        cfg.weights_init_type = WeightsInitType.xavier

        cfg.weights_init_kwargs = {'n_in': 0.1,'n_out': 0.1}
    elif n == 'normal':
        # training
        cfg.weights_init_type = WeightsInitType.normal

        cfg.weights_init_kwargs = {'sigma': 0.01}

cfg.gamma = 0.01
cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
cfg.nb_epoch = 100


