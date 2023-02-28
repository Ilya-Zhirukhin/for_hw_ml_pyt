from easydict import EasyDict
import numpy as np
cfg = EasyDict()
cfg.dataframe_path = 'C:\\Users\\Илья\\Desktop\\Ml_lib_students\\linear_regression_dataset.csv'

cfg.base_functions = [lambda x, p=i: x**p for i in range(1, 120+1)]
