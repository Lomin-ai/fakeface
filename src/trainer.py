import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import cv2
import numpy as np
from tqdm import tqdm, trange
from data.dataloader import get_trainval_loader, get_test_loader_real, get_test_loader_fake
from get_model import get_new_model, get_old_model
from util import compute_auroc
import copy

class Trainer():
    def __init__(self, cfg, checkpoint, logger):
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.logger = logger

        self.period_print = self.cfg.train.period_print
        self.period_val = self.cfg.train.period_val
        self.period_image = self.cfg.train.period_image
        self.period_checkpoint = self.cfg.train.period_checkpoint
        self.period_lr = self.cfg.train.period_lr

        self.loader_real_train, self.loader_real_val = get_trainval_loader(self.cfg, self.logger, 'real')
        self.loader_fake_train, self.loader_fake_val = get_trainval_loader(self.cfg, self.logger, 'fake')
        self.loader_real_test = get_test_loader_real(self.cfg, self.logger)
        self.loader_fake_test = get_test_loader_fake(self.cfg, self.logger)
        self.iter_loader_real_train = iter(self.loader_real_train)
        self.iter_loader_fake_train = iter(self.loader_fake_train)

        self.net = get_new_model(self.cfg)
        self.net = nn.DataParallel(self.net)
        self.net.cuda()
        self.epoch = 1
        self.iter = 0
       
        if self.cfg.model.finetune:
            parameters = self.net.module.last_linear.parameters()
        else:
            parameters = self.net.module.parameters()

        if self.cfg.train.optim == 'sgd':
            self.optim = optim.SGD(
                parameters,
                lr=self.cfg.train.lr_initial,
                momentum=self.cfg.train.momentum,
                nesterov=self.cfg.train.nesterov)
        elif self.cfg.train.optim == 'adam':
            self.optim = optim.Adam(
                parameters,
                lr=self.cfg.train.lr_initial,
                amsgrad=self.cfg.train.amsgrad)
        
        for i, group in enumerate(self.optim.param_groups):
            group.setdefault('initial_lr', group['lr'])
                    
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer = self.optim, 
            step_size = self.period_lr,
            gamma = 0.1,
            last_epoch = self.epoch)

        self.criterion = nn.CrossEntropyLoss()
        self.acc_real = 0
        self.acc_fake = 0
        self.num = 0
        self.loss = 0
        self.score = list()
        self.GT = list()
        self.epoch = 1

    def train(self):
        self.data_tic = time.time()
        self.data_time, self.train_time = 0, 0

        while True:
            self.lr_scheduler.step(self.epoch)
            self.lr = self.lr_scheduler.get_lr()

            self.iter += 1
            if self.iter > self.cfg.train.max_iter:
                break

            try:
                inp_real = next(self.iter_loader_real_train).cuda()
                inp_fake = next(self.iter_loader_fake_train).cuda()
            except StopIteration: # enables infinite loop regardless of dataset size
                self.epoch += 1
                self.loader_real_train.dataset.reset()
                self.loader_fake_train.dataset.reset()
                self.iter_loader_real_train = iter(self.loader_real_train)
                self.iter_loader_fake_train = iter(self.loader_fake_train)
                inp_real = next(self.iter_loader_real_train).cuda()
                inp_fake = next(self.iter_loader_real_train).cuda()

            tar_real = torch.zeros(self.cfg.train.batch_size // 2, dtype=torch.long).cuda()
            tar_fake = torch.ones(self.cfg.train.batch_size // 2, dtype=torch.long).cuda()

            self.data_time += time.time() - self.data_tic
            self.train_tic = time.time()

            inp = torch.cat((inp_real, inp_fake), 0)
            tar = torch.cat((tar_real, tar_fake), 0)

            self.net.train()
            out = self.net(inp)

            loss = self.criterion(out, tar)
            self.loss += loss.item()

            pred = torch.max(out, 1)[1]
            correct = torch.eq(pred, tar)
            num_half = inp.shape[0] // 2
            self.acc_real += correct[:num_half].sum().item()
            self.acc_fake += correct[num_half:].sum().item()
            self.num += num_half
            _out = out.clone().detach()
            self.score.extend(list(nn.functional.softmax(_out, 1)[:,1].cpu().numpy()))
            self.GT.extend(list(tar.cpu().numpy()))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.train_time += time.time() - self.train_tic

            self._snapshot(inp_real, inp_fake)

            if self.iter % self.period_val == 0:
                self.val()
                self.test()
            
            self.data_tic = time.time()

    @torch.no_grad()
    def val(self):
        tic_val = time.time()
        self.net.eval()

        pred = list()
        gt = list()

        score_real = 0
        acc_real = 0
        for i, inp in enumerate(self.loader_real_val):
            if i == 0:
                inp_real = inp
            output = self.net(inp.cuda()).cpu()
            score_real += nn.functional.softmax(output, 1)[:,0].sum().item()
            acc_real += torch.max(output, 1)[1].eq(0).sum().item()
            pred.extend(list(nn.functional.softmax(output, 1)[:,1].numpy()))

        score_fake = 0
        acc_fake = 0
        for i, inp in enumerate(self.loader_fake_val):
            if i == 0:
                inp_fake = inp
            output = self.net(inp.cuda()).cpu()
            score_fake += nn.functional.softmax(output, 1)[:,1].sum().item()
            acc_fake += torch.max(output, 1)[1].eq(1).sum().item()
            pred.extend(list(nn.functional.softmax(output, 1)[:,1].numpy()))

        gt.extend([0] * self.loader_real_val.dataset.num_face)
        gt.extend([1] * self.loader_fake_val.dataset.num_face)

        n_real = self.loader_real_val.dataset.num_face
        n_fake = self.loader_fake_val.dataset.num_face
        score_real /= max(n_real, 1)
        score_fake /= max(n_fake, 1)
        acc_real /= max(n_real, 1)
        acc_fake /= max(n_fake, 1)
        auroc, roc = compute_auroc(pred, gt)
        self.logger.write('')
        self.logger.write('Number of validation (real, fake) = ({}, {})'.format(n_real, n_fake))
        self.logger.write('[Validation average score] real: {:.2f} fake: {:.2f}'.format(score_real, score_fake))
        self.logger.write('[Validation accuracy]      real: {:.2f} fake: {:.2f}'.format(acc_real, acc_fake))
        self.logger.write('[Validation AUROC] {:.4f}'.format(auroc))
        self.logger.tensorboard.add_scalar('val/score/real', score_real, self.iter)
        self.logger.tensorboard.add_scalar('val/score/fake', score_fake, self.iter)
        self.logger.tensorboard.add_scalar('val/acc/real', acc_real, self.iter)
        self.logger.tensorboard.add_scalar('val/acc/fake', acc_fake, self.iter)
        self.logger.tensorboard.add_scalar('val/auroc', auroc, self.iter)
        img_real = inp_real[:,[2,1,0]] * self.cfg.dataset.std + self.cfg.dataset.mean
        img_fake = inp_fake[:,[2,1,0]] * self.cfg.dataset.std + self.cfg.dataset.mean
        self.logger.tensorboard.add_image('val/real', img_real, self.iter)
        self.logger.tensorboard.add_image('val/fake', img_fake, self.iter)

        self.logger.write('Time elapsed: {:.2f} sec\n'.format(time.time() - tic_val))

    @torch.no_grad()
    def test(self):
        tic_test = time.time()

        n_real = self.loader_real_test.dataset.num_face
        n_fake = np.array([_.dataset.num_face for _ in self.loader_fake_test]).sum()
        if n_real == 0 or n_fake == 0:
            return 
        
        self.net.eval()

        # Real
        score_real = 0
        acc_real = 0
        pred_real = list()
        gt_real = [0] * n_real
        for i, inp in enumerate(self.loader_real_test):
            if i == 0:
                inp_real = inp
            output = self.net(inp.cuda()).cpu()
            score_real += nn.functional.softmax(output, 1)[:,0].sum().item()
            acc_real += torch.max(output, 1)[1].eq(0).sum().item()
            pred_real.extend(list(nn.functional.softmax(output, 1)[:,1].numpy()))
        
        score_real /= max(n_real, 1)
        acc_real /= max(n_real, 1)

        # Fake
        for loader_fake_test in self.loader_fake_test:
            n_fake_this = loader_fake_test.dataset.num_face
            score_fake = 0
            acc_fake = 0
            pred_fake = list()
            gt_fake = [1] * n_fake_this
            for i, inp in enumerate(loader_fake_test):
                if i == 0:
                    inp_fake = inp
                output = self.net(inp.cuda()).cpu()
                score_fake += nn.functional.softmax(output, 1)[:,1].sum().item()
                acc_fake += torch.max(output, 1)[1].eq(1).sum().item()
                pred_fake.extend(list(nn.functional.softmax(output, 1)[:,1].numpy()))

            score_fake /= max(n_fake_this, 1)
            acc_fake /= max(n_fake_this, 1)

            pred = [_ for _ in pred_real] + pred_fake
            gt = [_ for _ in gt_real] + gt_fake
            auroc, roc = compute_auroc(pred, gt)

            fake_name = loader_fake_test.dataset.imdb.name
            self.logger.write(fake_name)
            self.logger.write('Number of test (real, fake) = ({}, {})'.format(n_real, n_fake_this))
            self.logger.write('[Test average score] real: {:.2f} fake: {:.2f}'.format(score_real, score_fake))
            self.logger.write('[Test accuracy]      real: {:.2f} fake: {:.2f}'.format(acc_real, acc_fake))
            self.logger.write('[Test AUROC] {:.4f}'.format(auroc))
            self.logger.tensorboard.add_scalar('test/score/real/' + fake_name, score_real, self.iter)
            self.logger.tensorboard.add_scalar('test/score/fake/' + fake_name, score_fake, self.iter)
            self.logger.tensorboard.add_scalar('test/acc/real/' + fake_name, acc_real, self.iter)
            self.logger.tensorboard.add_scalar('test/acc/fake/' + fake_name, acc_fake, self.iter)
            self.logger.tensorboard.add_scalar('test/auroc/' + fake_name, auroc, self.iter)
            img_real = inp_real[:,[2,1,0]] * self.cfg.dataset.std + self.cfg.dataset.mean
            img_fake = inp_fake[:,[2,1,0]] * self.cfg.dataset.std + self.cfg.dataset.mean
            self.logger.tensorboard.add_image('test/real/' + fake_name, img_real, self.iter)
            self.logger.tensorboard.add_image('test/fake/' + fake_name, img_fake, self.iter)

            self.logger.write('Time elapsed: {:.2f} sec\n'.format(time.time() - tic_test))

    def _snapshot(self, inp_real, inp_fake):
        if self.iter % self.period_print == 0:
            max_compute = 100
            # np.random.shuffle(self.score)
            # np.random.shuffle(self.GT)
            score = self.score[:max_compute]
            GT = self.GT[:max_compute]
            AUROC, ROC = compute_auroc(score, GT) 

            self.logger.write('Iter: {}, lr: {:.2e}, loss: {:.6f}, AUROC: {:.4f}, Time: {:.2f}sec ({:.2f}%)'.format(
                self.iter,
                self.lr[0],
                self.loss / self.num,
                AUROC,
                self.data_time + self.train_time,
                self.train_time * 100 / (self.data_time + self.train_time)))

            self.logger.tensorboard.add_scalar('lr', self.lr[0], self.iter)
            self.logger.tensorboard.add_scalar('train/loss', self.loss / self.num, self.iter)
            self.logger.tensorboard.add_scalar('train/acc_real', self.acc_real / self.num, self.iter)
            self.logger.tensorboard.add_scalar('train/acc_fake', self.acc_fake / self.num, self.iter)
            self.logger.tensorboard.add_scalar('train/AUROC', AUROC, self.iter)

            self.data_time, self.train_time = 0, 0
            self.acc_real, self.acc_fake = 0, 0
            self.loss, self.num = 0, 0
            self.score, self.GT = list(), list()

        if self.iter % self.period_checkpoint == 0:
            model_filename = '{}_epoch={:03d}_iter={:08d}.pkl'.format(
                self.cfg.tag, 
                self.epoch, 
                self.iter)
            model_path = os.path.join(self.checkpoint.model_path, model_filename)
            torch.save(copy.deepcopy(self.net.module).cpu().state_dict(), model_path)
        
        if self.iter % self.period_image == 0:
            img_filename = '{}_epoch={:03d}_iter={:08d}.jpg'.format(
                self.cfg.tag,
                self.epoch,
                self.iter)
            img_path = os.path.join(self.checkpoint.image_path, img_filename)
            img_real = inp_real[:,[2,1,0]] * self.cfg.dataset.std + self.cfg.dataset.mean
            img_fake = inp_fake[:,[2,1,0]] * self.cfg.dataset.std + self.cfg.dataset.mean
            torchvision.utils.save_image(img_real, img_path)
            self.logger.tensorboard.add_image('train/real', img_real, self.iter)
            self.logger.tensorboard.add_image('train/fake', img_fake, self.iter)
            