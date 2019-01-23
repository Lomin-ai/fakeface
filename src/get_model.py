import torch
import torch.nn as nn
from importlib import import_module

def get_old_model(cfg, checkpoint):
    model = get_new_model(cfg)

    assert checkpoint.last_model is not None
    last_model = checkpoint.last_model

    model.load_state_dict(last_model)
    return model


def get_new_model(cfg):
    modelname = cfg.model.name.lower().replace('-', '_').strip()

    if 'se_resnet' in modelname or 'senet' in modelname:
        from model import senet
        model = getattr(senet, modelname)()
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    elif 'se_resnext' in modelname:
        from model import senet
        model = getattr(senet, modelname)()
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    elif 'inceptionresnetv2' in modelname:
        from model import inceptionresnetv2
        model = getattr(inceptionresnetv2, modelname)()
        model.avgpool_1a = nn.AdaptiveAvgPool2d(1)

    elif 'nasnetalarge' in modelname:
        from model import nasnet 
        model = getattr(nasnet, modelname)(pretrained='imagenet+background')
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    elif 'pnasnet' in modelname:
        from model import pnasnet
        model = getattr(pnasnet, modelname)(pretrained='imagenet+background')
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    else:
        raise NotImplementedError()

    model.last_linear = nn.Linear(model.last_linear.in_features, 2)

    return model