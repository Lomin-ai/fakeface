import os
import numpy as np
import random
import math
import torch
import torch.utils.data as data 
from data import dataset 
from data import imdb
from data import dataloader
from data import faceswap
from torchvision import transforms
from PIL import Image
from tqdm import trange
import cv2
import traceback

class SizeError(Exception):
    pass


class OffsetError(Exception):
    pass


class DatasetBase(data.Dataset):

    def __init__(self, cfg, imdb, train_val_test, real_fake=None):
        self.cfg = cfg
        self.imdb = imdb
        self.real_fake = real_fake
        self.train_val_test = train_val_test
        self.resolution = cfg.dataset.resolution
        self.resolution_feed = cfg.dataset.resolution_feed

        if train_val_test == 'train':
            self._crop = self._crop_random
        elif train_val_test == 'val':
            self._crop = self._crop_center
        elif train_val_test == 'test':
            self._crop = self._crop_center
    
    def __getitem__(self, idx):

        raise NotImplementedError()

    def _flip(self, face):
        flip = transforms.RandomHorizontalFlip(0.5)
        face_PIL = Image.fromarray(face)
        face_PIL = flip(face_PIL)
        face = np.array(face_PIL)

        return face 

    def _normalize(self, face):
        face -= self.cfg.dataset.mean
        face /= self.cfg.dataset.std

        return face

    def _colorjitter(self, face):
        colorjitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.2)
        face_PIL = Image.fromarray(face)
        face_PIL = colorjitter(face_PIL)
        face = np.array(face_PIL)

        return face
    
    def _jpeg(self, face):
        quality = np.random.randint(60, 90)
        encoded = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        decoded = cv2.imdecode(encoded, -1)

        return decoded

    def _augment(self, face, idx):
        face = self._colorjitter(face)
        face = self._flip(face)
        if self.imdb.imgpath[idx].endswith('.png'):
            face = self._jpeg(face)
        
        return face

    def _read_img(self, idx):
        imgpath = os.path.join(self.cfg.data_path, self.imdb.imgpath[idx])
        img = cv2.imread(imgpath)

        return img

    def _crop_random(self, idx):
        t = random.betavariate(self.beta_a, self.beta_b)
        w = int((1 - t) * self.imdb.minsize[idx] + t * self.imdb.maxsize[idx] + 0.000001)
        h = int(w * self.cfg.dataset.ar + 0.000001)

        min_top, min_left, min_bottom, min_right = self.imdb.bbox_min[idx]
        max_top, max_left, max_bottom, max_right = self.imdb.bbox_max[idx]
        min_offset_x = max(max_left, min_right - w)
        min_offset_y = max(max_top, min_bottom - h)
        max_offset_x = min(max_right - w, min_left)
        max_offset_y = min(max_bottom - h, min_top)

        if min_offset_x > max_offset_x or min_offset_y > max_offset_y:
            raise OffsetError()

        offset_x = random.randrange(max_offset_x - min_offset_x + 1) + min_offset_x
        offset_y = random.randrange(max_offset_y - min_offset_y + 1) + min_offset_y
    
        return offset_y, offset_x, offset_y + h, offset_x + w

    def _crop_center(self, idx):
        average = (self.imdb.bbox_min[idx] + self.imdb.bbox_max[idx]) / 2.0

        return average.astype(np.int32)
    
    def _to_tensor(self, face):
        face = face.astype(np.float32) / 255
        face = self._normalize(face)
        face = face.transpose(2, 0, 1)
        face = torch.Tensor(face)

        return face

    def _default_get(self, idx):
        img = self._read_img(idx)
        top, left, bottom, right = self._crop(idx)
        face = img[top:bottom, left:right, :]
        face = cv2.resize(face, (self.resolution, self.resolution), cv2.INTER_CUBIC)
        face = cv2.resize(face, (self.resolution_feed, self.resolution_feed), cv2.INTER_CUBIC)

        return face
    
    def reset(self):
        self.imdb.reset()

    @property
    def num_face(self):

        return len(self.imdb.imgpath)

    def __len__(self):

        return self.num_face


class DatasetGAN(DatasetBase):

    def __init__(self, cfg, imdb, train_val_test, real_fake=None):
        super(DatasetGAN, self).__init__(cfg, imdb, train_val_test, real_fake)

        if self.cfg.dataset.beta_a > 0 and self.cfg.dataset.beta_b > 0:
            self.beta_a = self.cfg.dataset.beta_a
            self.beta_b = self.cfg.dataset.beta_b
        else:
            if self.resolution <= 64:
                self.beta_a = 0.8
                self.beta_b = 1.2
            elif self.resolution <= 128:
                self.beta_a = 1
                self.beta_b = 1.2
            else:
                self.beta_a = 1
                self.beta_b = 1.1

    def __getitem__(self, idx):
        idx = idx % self.num_face
        num_try = 0
        while True:
            try:
                face = self._default_get(idx)
                if self.train_val_test == 'train':
                    face = self._augment(face, idx)
                
                return self._to_tensor(face)

            except Exception:
                traceback.print_exc()
                idx = np.random.randint(self.num_face)
                num_try += 1
                if num_try > 10:
                    raise Exception()


class DatasetSyn(DatasetBase):

    def __init__(self, cfg, imdb, train_val_test, real_fake):
        super(DatasetSyn, self).__init__(cfg, imdb, train_val_test, real_fake)

        self.angle_range = cfg.dataset.angle_range
        self.lab_range = cfg.dataset.lab_range
        self.size_range = cfg.dataset.size_range
        self.face_keypoints = [
            list(range(1, 13)) + list(range(14, 17)) + list(range(22, 30)) + list(range(31, 35)), # eyes +  nose + cheeks
            list(range(14, 17)) + list(range(18, 22)) + list(range(26, 31)) + list(range(33,41)), #nose + mouth
            list(range(1, 13)) + list(range(22, 27)) + list(range(31, 33)), #eyes + eyebrows
            list(range(1, 13)) + list(range(14, 17)) + list(range(18, 41)), #eyes + nose + mouth
            list(range(1, 4)) + list(range(7, 10)) + list(range(14, 17)) + list(range(18, 21)) + [22, 24] + list(range(26, 32)) + list(range(33,41)), #left eye + nose + mouth
            list(range(4, 7)) + list(range(10, 13)) + list(range(14, 17)) + list(range(18, 21)) + [23] + list(range(25, 31)) + list(range(32,41)), #right eye + nose + mouth
            list(range(18, 21)) + list(range(29, 31))+ list(range(35,41)), #mouth
            list(range(1, 41)), #all
            list(range(1, 4)) + list(range(7, 10)) + [22, 24, 31], #left eye
            list(range(4, 7)) + list(range(10, 13)) + [23, 25, 32] #right eye
        ]

        self.swap_style = cfg.dataset.swap_style

        if self.cfg.dataset.beta_a > 0 and self.cfg.dataset.beta_b > 0:
            self.beta_a = self.cfg.dataset.beta_a
            self.beta_b = self.cfg.dataset.beta_b
        else:
            self.beta_a = 1
            self.beta_b = 1
    
    def reset(self):
        if hasattr(self.imdb, 'reset'):
            self.imdb.reset()
        self.swap_allowed_idx = np.where(np.not_equal(
            self.imdb.points, 
            np.zeros(self.imdb.points.shape)).all(2).all(1))[0]

    def _get_idx_dst(self, idx_src):
        if self.train_val_test == 'train':
            if hasattr(self, 'angle') and self.angle_range > 0:
                angle_src = self.angle[idx_src]
                mask_angle_low = angle_src - self.angle_range < self.angle
                mask_angle_high = self.angle < angle_src + self.angle_range
                mask_angle = mask_angle_low & mask_angle_high
            else:
                mask_angle = True
            
            if hasattr(self, 'lab') and self.lab_range > 0:
                lab_src = self.lab[idx_src]
                lab_diff = np.linalg.norm(0.000001 + lab_src - self.lab, axis=1)
                mask_lab = lab_diff < self.lab_range
            else:
                mask_lab = True

            if hasattr(self, 'imgsize') and self.size_range > 0:
                assert self.size_range > 0
                size_src = imgsize[idx_src].mean()
                mask_size_low = (1/self.size_range) * size_src < self.imgsize[:,0]
                mask_size_hight = self.imgsize[:,0] < self.size_range * size_src
                mask_size = mask_size_low
            else:
                mask_size = True

            mask = mask_angle & mask_lab & mask_size

            if (type(mask) != bool) and (mask.sum() < 2):
                raise Exception()

        else:
            mask = True

        if type(mask) == bool:
            cnt = 0
            while True:
                cnt += 1
                idx_dst = random.choice(self.swap_allowed_idx)
                if idx_dst != idx_src:
                    break
                if cnt > 10:
                    raise Exception()
        else:
            if mask.sum() < 2:
                raise Exception()

            idx_dst = random.randint(0, len(mask) - 1)
            idx_dst = mask.nonzero()[0][idx_dst]

        return idx_dst

    def _random_swap(self, idx_src, idx_dst, crop_rect):
        """ synthesize from src face into dst face
            crop_rect: the cropping face rect in dst image.
        """
        img_src = self._read_img(idx_src)
        img_dst = self._read_img(idx_dst)
        ann_src = self.imdb.points[idx_src]
        ann_dst = self.imdb.points[idx_dst]

        new_ann_src = faceswap.calc_new_keypoints(ann_src)
        new_ann_dst = faceswap.calc_new_keypoints(ann_dst)

        rect_src = tuple(self.imdb.bbox_original[idx_src])
        rect_dst = tuple(self.imdb.bbox_original[idx_dst])
        top, left, bottom, right = rect_dst
        rect_dst = (left, top, right - left, bottom - top)
        top, left, bottom, right = rect_src
        rect_src = (left, top, right - left, bottom - top)

        face_keyids = random.choice(self.face_keypoints)
        ann_src_part = [new_ann_src[keyid - 1] for keyid in face_keyids]
        ann_dst_part = [new_ann_dst[keyid - 1] for keyid in face_keyids]

        style = np.random.choice(list(self.swap_style.keys()), p=list(self.swap_style.values()))
        face, mask = faceswap.faceswap(
            img_src, rect_src, ann_src_part, 
            img_dst, rect_dst, ann_dst_part,
            style, crop_rect, self.resolution)

        # for debugging - KJH
        faceswap.draw_face_landmark(img_src, new_ann_src)
        img_src = cv2.rectangle(img_src,rect_src[:2], (rect_src[0]+rect_src[2], rect_src[1]+rect_src[3]),(0, 255, 0), 1)
        faceswap.draw_face_landmark(img_dst, new_ann_dst)
        img_dst = cv2.rectangle(img_dst,rect_dst[:2], (rect_dst[0]+rect_dst[2], rect_dst[1]+rect_dst[3]),(0, 255, 0), 1)
        cv2.imwrite("../../output/test/check_faceswap/UMDFaces/{}_{}_srcimg.jpg".format(idx_src,idx_dst), img_src)
        cv2.imwrite("../../output/test/check_faceswap/UMDFaces/{}_{}_dstimg.jpg".format(idx_src,idx_dst), img_dst)
        cv2.imwrite("../../output/test/check_faceswap/UMDFaces/{}_{}_face.jpg".format(idx_src,idx_dst), face)
        cv2.imwrite("../../output/test/check_faceswap/UMDFaces/{}_{}_mask.jpg".format(idx_src,idx_dst), mask)
        
        face = cv2.resize(face, (self.resolution_feed, self.resolution_feed), cv2.INTER_CUBIC)
        
        return face

    def __getitem__(self, idx):
        idx = idx % self.num_face
        num_try = 0
        while True:
            try:
                if (self.real_fake == 'fake') \
                    and (self.train_val_test == 'train' or self.train_val_test == 'val') \
                    and ('UMDFaces' in self.imdb.imgpath[idx]):
                    idx_src = idx
                    idx_dst = self._get_idx_dst(idx)
                    crop_rect = self._crop(idx_dst)
                    face = self._random_swap(idx_src, idx_dst, crop_rect)
                else:
                    face = self._default_get(idx)

                if self.train_val_test == 'train':
                    face = self._augment(face, idx)

                return self._to_tensor(face)

            except Exception:
                traceback.print_exc()
                idx = np.random.randint(self.num_face)
                num_try += 1
                if num_try > 10:
                    raise Exception()
    

class DatasetMod(DatasetGAN):
    pass
    
    
class DatasetGANSyn(DatasetSyn):
    pass

