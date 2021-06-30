# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


from model.CNNQLM_II import QA_quantum 
from model.vocab import CNNQLM_Vocab 
from model.dim   import CNNQLM_Dim
from model.CNNQLM_I   import CNNQLM_I 
from model.NNQLM_II   import NNQLM_II 
from model.NNQLM_I   import NNQLM_I
def setup(opt):
    
    if opt.model == 'QA_quantum':
        model = QA_quantum(opt)
    elif opt.model ==  'CNNQLM_I':
        model = CNNQLM_I(opt)
    elif opt.model == 'CNNQLM_Vocab' :
        model = CNNQLM_Vocab(opt)
    elif opt.model == 'CNNQLM_Dim' :
        model = CNNQLM_Dim(opt)
    elif opt.model == 'NNQLM_II' :
        model = NNQLM_II(opt)
    elif opt.model == 'NNQLM_I' :
        model = NNQLM_I(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model