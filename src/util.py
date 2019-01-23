import os
import shutil
import datetime
import yaml
from ast import literal_eval
from addict import Dict as addict
from tensorboardX import SummaryWriter


class Checkpoint():
    def __init__(self, cfg):
        self.cfg = cfg
        self.resume = False
        self.last_iter = 0
        self.last_model = None
        
        if cfg.tag is None:
            cfg.tag = cfg.now
        
        self._define_path() 

        if os.path.isdir(self.save_path):
            self._delete()

        self._make_path()

    def _define_path(self):
        self.log_path = os.path.join(self.cfg.cwd, self.cfg.log_path)
        self.save_path = os.path.join(self.log_path, self.cfg.tag)
        self.model_path = os.path.join(self.save_path, 'model')
        self.image_path = os.path.join(self.save_path, 'image')
        self.tensorboard_path = os.path.join(self.save_path, 'tensorboard')

    def _make_path(self):
        dirs = [self.log_path, \
                self.save_path, \
                self.model_path, \
                self.image_path, \
                self.tensorboard_path]
        for p in dirs:
            os.makedirs(p, exist_ok=True)

    def _delete(self):
        shutil.rmtree(self.save_path)


class Logger():
    def __init__(self, cfg, tensorboard_path):
        self.logFile = open(os.path.join(cfg.log_path, cfg.tag, 'log.txt'), 'a', 1)
        self.logFile.write(cfg.now + '\n')
        cfg_dict = cfg.to_dict()
        self.logFile.write('\n\n')
        self.logFile.write((yaml.dump(cfg_dict, default_flow_style=False)))
        self.logFile.write('\n\n')
        if tensorboard_path:
            self.tensorboard = SummaryWriter(tensorboard_path)

    def write(self, message):
        print(message)
        self.logFile.write(message + '\n')

    def close(self):
        self.logFile.write('\n\n\n')
        self.logFile.close()
        if hasattr(self, 'tensorboard'):
            self.tensorboard.close()


def from_list(cfg, li):
    assert len(li) % 2 == 0
    for k, v in zip(li[0::2], li[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]

        subkey = key_list[-1]
        assert subkey in d, 'Subkey \'{}\' is not found in configuration'.format(subkey)

        try:
            value = literal_eval(v)
        except:
            value = v

        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))

        d[subkey] = value
    
    return cfg


def get_config(args, now, src_dir):
    cfg = addict()
    cfg.now = now

    with open(os.path.join(src_dir, 'config/default.yaml'), 'r') as f:
        cfg_default = addict(yaml.load(f))
        cfg.update(cfg_default)

    with open(os.path.join(src_dir, 'config/{}.yaml'.format(args.preset.lower())), 'r') as f:
        cfg_preset = addict(yaml.load(f))
        cfg.update(cfg_preset)

    if args.set is not None:
        from_list(cfg, args.set)

    for arg in [_ for _ in dir(args) if not _.startswith('_') and _ is not 'set']:
        cfg[arg] = getattr(args, arg)

    paths = addict()
    paths.cwd = os.path.abspath(os.path.join(src_dir, '..'))
    paths.log_path = os.path.join(paths.cwd, cfg.log_path)
    paths.data_path = os.path.join(paths.cwd, cfg.data_path)
    paths.src_path = os.path.join(paths.cwd, cfg.src_path)
    paths.test_path = os.path.join(paths.cwd, cfg.test_path)
    paths.cache_path = os.path.join(paths.cwd, cfg.cache_path)
    cfg.update(paths)

    return cfg


def compute_auroc(predict, target, debug=False):
    """
    This function is to compute AUROC. 
    The dimension of two input parameters should be same.
    
    Parameters
    ----------
    predict : Prediction vector between 0 to 1 (single precision)
    target  : Label vector weather 0 or 1      (single precision or boolean)
    
    Returns
    -------
    ROC   : Positions of ROC curve (FPR, TPR)
            This is for plotting or validation purposes
    AUROC : The value of area under ROC curve
    """
    try:
        assert len(predict) > 0
        assert len(predict) == len(target), print('compute_auroc(): len(predict) != len(target)')

        n = len(predict)

        # Cutoffs are of prediction values
        cutoff = predict

        TPR = [0 for x in range(n+2)]
        FPR = [0 for x in range(n+2)]

        for k in range(n):
            predict_bin = 0

            TP = 0	# True	Positive
            FP = 0	# False Positive
            FN = 0	# False Negative
            TN = 0	# True	Negative

            for j in range(n):
                if (predict[j] >= cutoff[k]):
                    predict_bin = 1
                else :
                    predict_bin = 0

                TP = TP + (	 predict_bin  &      target[j]	) 
                FP = FP + (	 predict_bin  & (not target[j]) )
                FN = FN + ( (not predict_bin) &	 target[j]	)
                TN = TN + ( (not predict_bin) & (not target[j]) )

            # True	Positive Rate
            TPR[k] = float(TP) / float(TP + FN)
            # False Positive Rate
            FPR[k] = float(FP) / float(FP + TN)

        TPR[n] = 0.0
        FPR[n] = 0.0
        TPR[n+1] = 1.0
        FPR[n+1] = 1.0
       
        # Positions of ROC curve (FPR, TPR)
        ROC = sorted(zip(FPR, TPR), reverse=True)

        AUROC = 0

        # Compute AUROC Using Trapezoidal Rule
        for j in range(n+1):
            h =   ROC[j][0] - ROC[j+1][0]
            w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

            AUROC = AUROC + h*w

        return AUROC, ROC

    except Exception as e:
        print(e)
        return 0, 0