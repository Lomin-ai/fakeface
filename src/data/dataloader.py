import math
import numpy as np
import torch
import torch.utils.data as data 
from data import imdb
from data import dataset

def get_trainval_loader(cfg, logger, real_fake):
    imdb_train = imdb.Union(cfg, 'imdb_train', virtual=True, logger=logger, train=True)
    imdb_val = imdb.Union(cfg, 'imdb_val', virtual=True, logger=logger, train=False)

    if cfg.dataset.minimal:
        dsets = {name: weight for name, weight in cfg.dataset.minimal_set[real_fake].items()}
        assert np.array(list(dsets.values())).min() > 0.0
    else:
        if real_fake == 'real':
            dsets = {name: weight for name, weight in cfg.dataset.real_set.items() if weight > 0}
        elif real_fake == 'fake':
            dsets = {name: weight for name, weight in cfg.dataset.fake_set.items() if weight > 0}

    assert math.isclose(np.array(list(dsets.values())).sum(), 1.0)

    for dset_name, weight in dsets.items():
        class_name = dset_name.replace('-', '_')

        _imdb = getattr(imdb, class_name)(cfg, name=dset_name, virtual=False, logger=logger)

        assert dset_name in cfg.dataset.num_val
        num_val = cfg.dataset.num_val[dset_name]
        num_test = cfg.dataset.num_test[dset_name] if dset_name in cfg.dataset.num_test else 0

        _imdb_trainval, _imdb_test = _imdb.split(num_test)
        _imdb_train, _imdb_val = _imdb_trainval.split(num_val)

        imdb_train.merge(_imdb_train, weight)
        imdb_val.merge(_imdb_val)

    imdb_train.initialize()
    imdb_val.initialize()

    dataset_class = {
        'gan': dataset.DatasetGAN,
        'syn': dataset.DatasetSyn,
        'mod': dataset.DatasetMod,
        'gan+syn': dataset.DatasetGANSyn
    }
    _dataset = dataset_class[cfg.preset]
    dataset_train = _dataset(cfg, imdb_train, 'train', real_fake)
    dataset_val = _dataset(cfg, imdb_val, 'val', real_fake)
    
    dataset_train.reset()
    dataset_val.reset()

    assert dataset_train.num_face > 0

    dataloader_train = torch.utils.data.DataLoader(
        dataset = dataset_train,
        batch_size = cfg.train.batch_size // 2,
        shuffle = cfg.train.data_loader.shuffle,
        drop_last = cfg.train.data_loader.drop_last,
        num_workers = cfg.train.data_loader.num_workers,
        pin_memory = cfg.train.data_loader.pin_memory 
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset = dataset_val,
        batch_size = cfg.train.batch_size_val,
        shuffle = False,
        drop_last = False,
        num_workers = cfg.train.data_loader.num_workers,
        pin_memory = cfg.train.data_loader.pin_memory 
    )

    return dataloader_train, dataloader_val

def get_test_loader_real(cfg, logger):
    _imdb = imdb.Sample_1_M2_Real(cfg, name='Sample_1_M2_Real', virtual=False, logger=logger)
    _dataset = dataset.DatasetGAN(cfg, _imdb, 'test', 'real')

    dataloader_test = torch.utils.data.DataLoader(
            dataset = _dataset,
            batch_size = cfg.train.batch_size_test,
            shuffle = True,
            drop_last = False,
            num_workers = cfg.train.data_loader.num_workers,
            pin_memory = cfg.train.data_loader.pin_memory
    )

    return dataloader_test

def get_test_loader_fake(cfg, logger):
    datasets = list()

    if cfg.preset == 'gan':
        _imdb_1 = imdb.Sample_1_M1_GAN(cfg, name='Sample_1_M1_GAN', virtual=False, logger=logger)
        _imdb_2 = imdb.Sample_2_GAN(cfg, name='Sample_2_GAN', virtual=False, logger=logger)
        datasets.append(dataset.DatasetGAN(cfg, _imdb_1, 'test', 'fake'))
        datasets.append(dataset.DatasetGAN(cfg, _imdb_2, 'test', 'fake'))

    elif cfg.preset == 'syn':
        _imdb = imdb.Sample_2_Syn(cfg, name='Sample_2_Syn', virtual=False, logger=logger)
        datasets.append(dataset.DatasetSyn(cfg, _imdb, 'test', 'fake'))

    elif cfg.preset == 'mod':
        if cfg.dataset.minimal:
            dsets = {name: weight for name, weight in cfg.dataset.minimal_set[real_fake].items()}
            assert np.array(list(dsets.values())).min() > 0.0
        else:
            dsets = {name: weight for name, weight in cfg.dataset.fake_set.items() if weight > 0}

        for dset_name, weight in dsets.items():
            class_name = dset_name.replace('-', '_')
            assert 'Glow' in class_name or 'StarGAN' in class_name
            _imdb = getattr(imdb, class_name)(cfg, name=dset_name, virtual=False, logger=logger, test=True)
            datasets.append(dataset.DatasetMod(cfg, _imdb, 'test', 'fake'))

    elif cfg.preset == 'gan+syn':
        _imdb_gan = imdb.Sample_2_GAN(cfg, name='Sample_2_GAN', virtual=False, logger=logger)
        _imdb_syn = imdb.Sample_2_Syn(cfg, name='Sample_2_Syn', virtual=False, logger=logger)
        datasets.append(dataset.DatasetGANSyn(cfg, _imdb_gan, 'test', 'fake'))
        datasets.append(dataset.DatasetGANSyn(cfg, _imdb_syn, 'test', 'fake'))
        _imdb_all = imdb.Union(cfg, name='Sample_2_GAN+Syn', virtual=True, logger=logger, train=False)
        _imdb_all.merge(_imdb_gan)
        _imdb_all.merge(_imdb_syn)
        _imdb_all.initialize()
        _dataset = dataset.DatasetGANSyn(cfg, _imdb_all, 'test', 'fake')
        _dataset.reset()
        datasets.append(_dataset)

    dataloaders_test = list()
    for _dataset in datasets:
        dataloaders_test.append(torch.utils.data.DataLoader(
                dataset = _dataset,
                batch_size = cfg.train.batch_size_test,
                shuffle = True,
                drop_last = False,
                num_workers = cfg.train.data_loader.num_workers,
                pin_memory = cfg.train.data_loader.pin_memory
        ))

    return dataloaders_test