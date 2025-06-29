import sys
import os

import tensorflow as tfw





def create_model(*args,**kwargs):
        
    sys.path.append("/media/maxime/Crucial X8/lczero_create_bt4/lczero-training/")
    sys.path.append("/media/maxime/Crucial X8/lczero_create_bt4/lczero-training/ta")
    from ta.tfprocess import TFProcess

    import yaml
    with open("/media/maxime/Crucial X8/lczero_create_bt4/lczero-training/BT4.yaml") as file:
            cfg = yaml.safe_load(file)
    tfp = TFProcess(cfg)
    tfp.init_net()
    bt4 = tfp.model
    bt4.load_weights("/media/maxime/Crucial X8/lczero_create_bt4/lczero-training/true_bt4_2.h5")
    return bt4