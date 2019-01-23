import os
import pickle
import numpy as np
from tqdm import trange
from glob import glob
import re
import csv
import math

class Imdb(object):

    def __init__(self, cfg, name=None, virtual=False, logger=None, **kwargs):
        self.cache_attr = [
            'imgpath', 'imgsize', 
            'bbox_original', 'bbox_min', 'bbox_max', 
            'minsize', 'maxsize',
            'angle', 'points', 'lab',
            'partname', 'level', 'bbox_mask'
        ]
        self.cfg = cfg
        self.name = name 
        self.virtual = virtual
        self.logger = logger
        self.cache_filename = os.path.join(cfg.cache_path, '{}.pkl'.format(name))
        assert self.logger

        self._set(kwargs)

        if not virtual:
            try:
                assert not self.cfg.clear_cache
                cache = self._check_cache(cfg)
                for attr in self.cache_attr:
                    if attr in cache:
                        setattr(self, attr, cache[attr])
            except:
                self._parse_annotation()
                self._set_bbox_range()
                self._bbox_sanity()
                self._set_minmax()
                self._save_cache()

            self._filter()
            self._order()

        else:
            self.imgpath = np.empty(0)
            self.imgsize = np.empty((0, 2), dtype=np.int32)
            self.bbox_original = np.empty((0, 4), dtype=np.int32)
            self.bbox_min = np.empty((0, 4), dtype=np.int32)
            self.bbox_max = np.empty((0, 4), dtype=np.int32)
            self.minsize = np.empty(0, dtype=np.int32)
            self.maxsize = np.empty(0, dtype=np.int32)
            # For UMDFaces
            self.angle = np.empty((0, 3), dtype=np.float32) # yaw, pitch, roll
            self.points = np.empty((0, 21, 2), dtype=np.int32) # 21 points
            self.lab = np.empty((0, 3), dtype=np.float32) # mean LAB of face region
            # For Photoshop
            self.partname = np.empty(0)
            self.level = np.empty(0, dtype=np.int32)
            self.bbox_mask = np.empty((0, 4), dtype=np.int32)

    def _set(self, kwargs):
        self.ignore_cache = False 

    def _parse_annotation(self):
        raise NotImplementedError()

    def _set_bbox_range(self):
        raise NotImplementedError()

    def _bbox_sanity(self):
        self.bbox_min[:,:2] = np.maximum(0, self.bbox_min[:,:2]) # top, left (minbox)
        self.bbox_max[:,:2] = np.maximum(0, self.bbox_max[:,:2]) # top, left (maxbox)

        if hasattr(self, 'imgsize'):
            self.bbox_min[:,2] = np.minimum(self.imgsize[:,0], self.bbox_min[:,2]) # bottom
            self.bbox_max[:,2] = np.minimum(self.imgsize[:,0], self.bbox_max[:,2]) # bottom
            self.bbox_min[:,3] = np.minimum(self.imgsize[:,1], self.bbox_min[:,3]) # right
            self.bbox_max[:,3] = np.minimum(self.imgsize[:,1], self.bbox_max[:,3]) # right

    def _set_minmax(self):
        min_width = self.bbox_min[:,3] - self.bbox_min[:,1]
        min_height = self.bbox_min[:,2] - self.bbox_min[:,0]
        max_width = self.bbox_max[:,3] - self.bbox_max[:,1]
        max_height = self.bbox_max[:,2] - self.bbox_max[:,0]

        self.minsize = np.maximum(min_width, (min_height / self.cfg.dataset.ar).astype(np.int32))
        self.maxsize = np.minimum(max_width, (max_height / self.cfg.dataset.ar).astype(np.int32))

    def _filter(self):
        reso_th = self.cfg.dataset.resolution_thres_factor * self.cfg.dataset.resolution
        reso_th = max(self.cfg.dataset.min_original_resolution, reso_th)
        mask_w = (self.bbox_max[:,3] - self.bbox_max[:,1]) > reso_th
        mask_h = (self.bbox_max[:,2] - self.bbox_max[:,0]) > reso_th * self.cfg.dataset.ar
        mask_minmax = self.minsize <= self.maxsize

        mask = mask_w & mask_h & mask_minmax 
        if hasattr(self, 'angle'):
            angle_th = self.cfg.dataset.angle_th
            mask_angle = (-angle_th < self.angle) & (self.angle < angle_th)
            mask_angle = mask_angle.all(1)
            mask = mask & mask_angle
        
        self.imgpath = self.imgpath[mask]
        self.bbox_original = self.bbox_original[mask]
        self.bbox_min = self.bbox_min[mask]
        self.bbox_max = self.bbox_max[mask]
        self.minsize = self.minsize[mask]
        self.maxsize = self.maxsize[mask]
        if hasattr(self, 'imgsize'):
            self.imgsize = self.imgsize[mask]
        if hasattr(self, 'angle'):
            self.angle = self.angle[mask]
        if hasattr(self, 'points'):
            self.points = self.points[mask]
        if hasattr(self, 'lab'):
            self.lab = self.lab[mask]
        if hasattr(self, 'partname'):
            self.partname = self.partname[mask]
        if hasattr(self, 'level'):
            self.level = self.level[mask]
        if hasattr(self, 'bbox_mask'):
            self.bbox_mask = self.bbox_mask[mask]

    def _order(self):
        idx = np.argsort(self.imgpath, axis=0)
        self.imgpath = self.imgpath[idx]
        self.bbox_original = self.bbox_original[idx]
        self.bbox_min = self.bbox_min[idx]
        self.bbox_max = self.bbox_max[idx]
        self.minsize = self.minsize[idx]
        self.maxsize = self.maxsize[idx]
        if hasattr(self, 'imgsize'):
            self.imgsize = self.imgsize[idx]
        if hasattr(self, 'angle'):
            self.angle = self.angle[idx]
        if hasattr(self, 'points'):
            self.points = self.points[idx]
        if hasattr(self, 'lab'):
            self.lab = self.lab[idx]
        if hasattr(self, 'partname'):
            self.partname = self.partname[idx]
        if hasattr(self, 'level'):
            self.level = self.level[idx]
        if hasattr(self, 'bbox_mask'):
            self.bbox_mask = self.bbox_mask[idx]
    
    def split(self, num_2):
        imdb_1 = Imdb(self.cfg, self.name, virtual=True, logger=self.logger)
        imdb_2 = Imdb(self.cfg, self.name, virtual=True, logger=self.logger)
        # num_val = self.cfg.dataset.num_val[self.name]
        num_1 = len(self.imgpath) - num_2

        imdb_1.imgpath = self.imgpath[:num_1]
        imdb_1.bbox_original = self.bbox_original[:num_1]
        imdb_1.bbox_min = self.bbox_min[:num_1]
        imdb_1.bbox_max = self.bbox_max[:num_1]
        imdb_1.minsize = self.minsize[:num_1]
        imdb_1.maxsize = self.maxsize[:num_1]

        imdb_2.imgpath = self.imgpath[num_1:]
        imdb_2.bbox_original = self.bbox_original[num_1:]
        imdb_2.bbox_min = self.bbox_min[num_1:]
        imdb_2.bbox_max = self.bbox_max[num_1:]
        imdb_2.minsize = self.minsize[num_1:]
        imdb_2.maxsize = self.maxsize[num_1:]

        if hasattr(self, 'imgsize'):
            imdb_1.imgsize = self.imgsize[:num_1]
            imdb_2.imgsize = self.imgsize[num_1:]

        if hasattr(self, 'angle'):
            imdb_1.angle = self.angle[:num_1]
            imdb_2.angle = self.angle[num_1:]

        if hasattr(self, 'points'):
            imdb_1.points = self.points[:num_1]
            imdb_2.points = self.points[num_1:]
        
        if hasattr(self, 'lab'):
            imdb_1.lab = self.lab[:num_1]
            imdb_2.lab = self.lab[num_1:]

        if hasattr(self, 'partname'):
            imdb_1.partname = self.partname[:num_1]
            imdb_2.partname = self.partname[num_1:]

        if hasattr(self, 'level'):
            imdb_1.level = self.level[:num_1]
            imdb_2.level = self.level[num_1:]

        if hasattr(self, 'bbox_mask'):
            imdb_1.bbox_mask = self.bbox_mask[:num_1]
            imdb_2.bbox_mask = self.bbox_mask[num_1:]

        return imdb_1, imdb_2 

    def _check_cache(self, cfg):
        assert not self.ignore_cache
        with open(self.cache_filename, 'rb') as f:
            cache = pickle.load(f)
            return cache

    def _save_cache(self):
        os.makedirs(self.cfg.cache_path, exist_ok=True)
        with open(self.cache_filename, 'wb') as f:
            cache_data = {}
            for attr in self.cache_attr:
                if hasattr(self, attr):
                    cache_data[attr] = getattr(self, attr)
            pickle.dump(cache_data, f)


class Union(Imdb):

    def _set(self, kwargs):
        self.train = kwargs['train']
        self.imdbs = dict()
        self.weights = dict()
        self.total = 0

    def merge(self, imdb, weight=0.0):
        assert self.virtual
        self.imdbs[imdb.name] = imdb
        self.weights[imdb.name] = weight

    def initialize(self):
        if self.train:
            total_candidates = [
                int(len(self.imdbs[imdb_name].imgpath) / self.weights[imdb_name]) \
                    for imdb_name in self.imdbs.keys()]
            self.total = np.array(total_candidates).min()
        else:
            self.total = 0

    def reset(self):
        self.imgpath = np.empty(0)
        self.imgsize = np.empty((0, 2), dtype=np.int32)
        self.bbox_original = np.empty((0, 4), dtype=np.int32)
        self.bbox_min = np.empty((0, 4), dtype=np.int32)
        self.bbox_max = np.empty((0, 4), dtype=np.int32)
        self.minsize = np.empty(0, dtype=np.int32)
        self.maxsize = np.empty(0, dtype=np.int32)
        # For UMDFaces
        self.angle = np.empty((0, 3), dtype=np.float32) # yaw, pitch, roll
        self.points = np.empty((0, 21, 2), dtype=np.int32) # 21 points
        self.lab = np.empty((0, 3), dtype=np.float32) # mean LAB of face region
        # For Photoshop
        self.partname = np.empty(0)
        self.level = np.empty(0, dtype=np.int32)
        self.bbox_mask = np.empty((0, 4), dtype=np.int32)

        for imdb_name in self.imdbs.keys():
            imdb = self.imdbs[imdb_name]
            length = len(imdb.imgpath)
            weight = self.weights[imdb_name]

            if self.train:
                num_select = int(self.total * weight)
                select = np.random.choice(length, size=num_select, replace=False)
            else:
                select = np.arange(length)
                num_select = len(select)

            self.imgpath = np.append(self.imgpath, imdb.imgpath[select], axis=0)
            self.bbox_original = np.append(self.bbox_original, imdb.bbox_original[select], axis=0)
            self.bbox_min = np.append(self.bbox_min, imdb.bbox_min[select], axis=0)
            self.bbox_max = np.append(self.bbox_max, imdb.bbox_max[select], axis=0)
            self.minsize = np.append(self.minsize, imdb.minsize[select], axis=0)
            self.maxsize = np.append(self.maxsize, imdb.maxsize[select], axis=0)

            if hasattr(imdb, 'imgsize') and len(imdb.imgsize) == length:
                self.imgsize = np.append(self.imgsize, imdb.imgsize[select], axis=0)
            else:
                self.imgsize = np.append(self.imgsize, np.zeros((num_select, 2)), axis=0)

            if hasattr(imdb, 'angle') and len(imdb.angle) == length:
                self.angle = np.append(self.angle, imdb.angle[select], axis=0)
            else:
                self.angle = np.append(self.angle, np.ones((num_select, 3)) * (-360), axis=0)
            
            if hasattr(imdb, 'points') and len(imdb.points) == length:
                self.points = np.append(self.points, imdb.points[select], axis=0)
            else:
                self.points = np.append(self.points, np.zeros((num_select, 21, 2)), axis=0)
            
            if hasattr(imdb, 'lab') and len(imdb.lab) == length:
                self.lab = np.append(self.lab, imdb.lab[select], axis=0)
            else:
                self.lab = np.append(self.lab, np.zeros((num_select, 3)), axis=0)


class AFLW(Imdb):

    def _set(self, kwargs):
        raise NotImplementedError()

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, self.name)

        import sqlite3 as sq
        aflw_sq_path = os.path.join(root, 'data/aflw.sqlite') 
        aflw_sq = sq.connect(aflw_sq_path)
        aflw_cur = aflw_sq.cursor()
        aflw_cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_name = aflw_cur.fetchall()

        # fetch image_name, face_rect and feature coordinates from db# fetch i 
        faces = aflw_cur.execute("SELECT * FROM Faces")
        face_ids = faces.fetchall()

        face_names = []
        face_rects = []
        face_features = []
        imgpath = list()
        bbox = list()
        for i in range(len(face_ids)): 
            # get face_id and file_id
            face_id = face_ids[i][0]
            file_id_sqlite = "SELECT file_id FROM Faces WHERE face_id ='" + str(face_id) + "'"
            file_id = aflw_cur.execute(file_id_sqlite).fetchall()
            file_id = file_id[0][0] # 'img00035.jpg'
            if len(file_id) < 1:
                continue
            
            # get file_path
            face_name_query = "SELECT filepath FROM FaceImages WHERE file_id = '"+ file_id + "'"
            face_name = aflw_cur.execute(face_name_query).fetchall()
            face_name = face_name[0][0] # '3/image00035.jpg'

            # rect
            feature_rect_query = "SELECT FaceRect.x,FaceRect.y,FaceRect.w,FaceRect.h FROM FaceRect WHERE face_id ='" + str(face_id) + "'"
            feature_rect = aflw_cur.execute(feature_rect_query).fetchall() # [(62, 64, 348, 348)]
            if len(feature_rect) < 1:
                continue
            
            feature_rect = feature_rect[0]
            x = feature_rect[0]
            y = feature_rect[1]
            w = feature_rect[2]
            h = feature_rect[3]
            
            # coor (normalize to 0~1)
            feature_coor_query = "SELECT descr,FeatureCoords.x,FeatureCoords.y FROM FeatureCoords,FeatureCoordTypes WHERE face_id ='" + str(face_id) + "' AND FeatureCoords.feature_id = FeatureCoordTypes.feature_id"
            feature_coor = aflw_cur.execute(feature_coor_query).fetchall()    
            coor_x = [-1 for k in range(5)]
            coor_y = [-1 for k in range(5)]
            for j in range(len(feature_coor)):
                if feature_coor[j][0] == 'LeftEyeCenter':
                    coor_x[0] = feature_coor[j][1]
                    coor_y[0] = feature_coor[j][2]
                elif feature_coor[j][0] == 'RightEyeCenter':
                    coor_x[1] = feature_coor[j][1]
                    coor_y[1] = feature_coor[j][2]
                elif feature_coor[j][0] == 'NoseCenter':
                    coor_x[2] = feature_coor[j][1]
                    coor_y[2] = feature_coor[j][2]
                elif feature_coor[j][0] == 'MouthLeftCorner':
                    coor_x[3] = feature_coor[j][1]
                    coor_y[3] = feature_coor[j][2]
                elif feature_coor[j][0] == 'MouthRightCorner':
                    coor_x[4] = feature_coor[j][1]
                    coor_y[4] = feature_coor[j][2]
            
            coor = []
            coor.append(coor_x)
            coor.append(coor_y)

            imgpath.append(os.path.join(self.name, 'aflw/data/flickr', face_name))
            bbox.append([y, x, y + h, x + w])
            # self.coor.append(coor)

        aflw_cur.close()
        aflw_sq.close()

        self.imgpath = np.array(imgpath)
        self.bbox_original = np.array(bbox, dtype=np.int32)

    def _set_bbox_range(self):
        top, left, bottom, right = np.split(self.bbox_original, np.arange(1, 4), axis=1)
        h = bottom - top
        w = right - left
        size_min = np.max((h, w), 0) * 1.2
        size_max = np.max((h, w), 0) * 1.4
        cx = np.mean((left, right), 0)
        cy = np.mean((top, bottom), 0)
        cx_min = cx
        cx_max = cx
        cy_min = cy - 0.15 * size_min
        cy_max = cy - 0.15 * size_max

        self.bbox_min = np.hstack((
            cy_min - 0.5 * size_min,
            cx_min - 0.4 * size_min,
            cy_min + 0.5 * size_min,
            cx_min + 0.4 * size_min)).astype(np.int32)

        self.bbox_max = np.hstack((
            cy_max - 0.5 * size_max,
            cx_max - 0.5 * size_max,
            cy_max + 0.5 * size_max,
            cx_max + 0.5 * size_max)).astype(np.int32)


class CelebA(Imdb):
    
    def _set(self, kwargs):
        raise NotImplementedError()

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, self.name)
        anno_file = open(os.path.join(root, 'Anno/list_bbox_celeba.txt'), 'r')
        lines = anno_file.readlines()

        for line in lines[2:]:
            assert '.jpg' in line
            elem = list(filter(lambda a: a != '', line.strip().split(' ')))
            imgname = elem[0]
            left, top, width, height = tuple(elem[1:])
            left = int(left)
            top = int(top)
            width = int(width)
            height = int(height)
            bbox = [top, left, top + height, left + width]

            relpath = os.path.join(self.name, 'Img/img_celeba')
            self.imgpath.append(os.path.join(relpath, imgname))
            self.bbox_original.append(bbox)

        anno_file.close()
    
    def _set_bbox_range(self):
        for bb_o in self.bbox_original:
            top, left, bottom, right = bb_o
            h = bottom - top
            w = right - left

            center_x = (left + right) / 2
            w_min = 0.8 * w
            top_min = int(top + 0.1 * h)
            left_min = int(center_x - w_min / 2)
            bottom_min = int(bottom - 0.25 * h)
            right_min = int(center_x + w_min / 2)
            center_y = (top_min + bottom_min) / 2

            h_new = 2 * (bottom_min - top_min)
            w_new = 2 * (right_min - left_min)

            top_pad = int(center_y - h_new / 2)
            left_pad = int(center_x - w_new / 2)
            bottom_pad = int(center_y + h_new / 2)
            right_pad = int(center_x + w_new / 2)

            bbox_min = [top_min, left_min, bottom_min, right_min]
            bbox_max = [top_pad, left_pad, bottom_pad, right_pad]
            self.bbox_min.append(bbox_min)
            self.bbox_max.append(bbox_max)


class CelebA_HQ(Imdb):

    def _set(self, kwargs):
        self.num_img = 30 * 1000

    def _parse_annotation(self):
        imgpath = list()
        imgsize = list()
        bbox = list()
        for i in range(self.num_img):
            imgname = '{:06d}.png'.format(i)
            imgpath.append(os.path.join(self.name, imgname))
            imgsize.append([1024, 1024])
            bbox.append([0, 0, 1024, 1024])
        
        self.imgpath = np.array(imgpath)
        self.imgsize = np.array(imgsize, dtype=np.int32)
        self.bbox_original = np.array(bbox, dtype=np.int32)
    
    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)
        # offset_x = 212
        # offset_y = 170
        # min_size_x = 600
        # min_size_y = 650

        # top = offset_y
        # bottom = top + min_size_y
        # left = offset_x
        # right = left + min_size_x

        # length = len(self.bbox_original)
        # self.bbox_min = np.array([[top, left, bottom, right]] * length, dtype=np.int32)
        # self.bbox_max = np.array([[0, 0, 1024, 1024]] * length, dtype=np.int32)


class IJB_C(Imdb):

    def _set(self, kwargs):
        raise NotImplementedError()

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, self.name)

        anno_file = open(os.path.join(root, 'protocols/ijbc_face_detection_ground_truth.csv'))
        lines = anno_file.readlines()

        for line in lines[1:]:
            imgpath, left, top, width, height, ignore = tuple(line.strip().split(','))
            if imgpath.split('/')[0] != 'nonfaces':
                left = int(left)
                top = int(top)
                width = int(width)
                height = int(height)
                ignore = int(ignore)
                imgpath = os.path.join(self.name, 'images', imgpath)
                bbox = [top, left, top + height, left + width]
                self.imgpath.append(imgpath)
                self.bbox_original.append(bbox)
                # self.ignore.append(ignore)

        anno_file.close()

    def _set_bbox_range(self):
        for bb_o in self.bbox_original:
            top, left, bottom, right = bb_o
            center_y = (top + bottom) / 2
            center_x = (left + right) / 2

            h = bottom - top
            w = right - left
            # top_min = int(top)
            # left_min = int(left + 0.1 * w)
            # bottom_min = int(bottom)
            # right_min = int(right - 0.1 * w)
            h_new = 2 * h
            w_new = 2 * w

            top_pad = int(center_y - h_new / 2)
            left_pad = int(center_x - w_new / 2)
            bottom_pad = int(center_y + h_new / 2)
            right_pad = int(center_x + w_new / 2)

            # bbox_min = [top_min, left_min, bottom_min, right_min]
            bbox_min = bb_o
            bbox_max = [top_pad, left_pad, bottom_pad, right_pad]
            self.bbox_min.append(bbox_min)
            self.bbox_max.append(bbox_max)


class UMDFaces(Imdb):
    
    def _parse_annotation(self):
        with open(os.path.join(self.cfg.cache_path, 'UMDFaces_img.csv')) as f:
            imgpath = list()
            imgsize = list()
            bbox = list()
            angle = list() # yaw, pitch, roll
            points = list() # 21 points
            lab = list()

            rdr = csv.DictReader(f)
            for row in rdr:
                imgpath.append(os.path.join(self.name, 'umdfaces_batch{}'.format(int(row['fold'])), row['FILE']))
                imgsize.append([int(row['img_h']), int(row['img_w'])])
                bbox.append([
                    float(row['FACE_Y']),
                    float(row['FACE_X']),
                    float(row['FACE_Y']) + float(row['FACE_HEIGHT']),
                    float(row['FACE_X']) + float(row['FACE_WIDTH'])])
                angle.append([
                    float(row['YAW']),
                    float(row['PITCH']),
                    float(row['ROLL'])])
                pts = []
                for i in range(21):
                    pts.append([
                        float(row['P{}X'.format(i + 1)]),
                        float(row['P{}Y'.format(i + 1)])
                        # int(float(row['VIS{}'.format(i + 1)]))
                    ])
                points.append(pts)
                lab.append([
                    float(row['meanL']) * 100 / 255,
                    float(row['meanA']) - 128,
                    float(row['meanB']) - 128])

        self.imgpath = np.array(imgpath)
        self.imgsize = np.array(imgsize, dtype=np.int32)
        self.bbox_original = np.array(bbox, dtype=np.int32)
        self.angle = np.array(angle, dtype=np.float32)
        self.points = np.array(points, dtype=np.int32)
        self.lab = np.array(lab, dtype=np.float32)

    def _set_bbox_range(self):
        top, left, bottom, right = np.split(self.bbox_original, np.arange(1, 4), axis=1)
        h = bottom - top
        w = right - left
        size_min = np.max((h, w), 0) * 1.3
        size_max = np.max((h, w), 0) * 1.7
        cx = np.mean((left, right), 0)
        cy = np.mean((top, bottom), 0)

        th_x, th_y = 0.2, 0.2
        yaw, pitch, roll = np.split(self.angle, 3, axis=1)
        # x
        cx_delta_min = 0.3 * np.sin(np.radians(yaw))
        cx_delta_max = 0.3 * np.sin(np.radians(yaw))
        cx_min = cx - w * np.clip(cx_delta_min, -th_x, th_x)
        cx_max = cx - w * np.clip(cx_delta_max, -th_x, th_x)
        # y
        cy_delta_min = 0.1 * h * np.sin(np.radians(pitch)) + 0.15 * size_min
        cy_delta_max = 0.1 * h * np.sin(np.radians(pitch)) + 0.15 * size_max
        cy_min = cy - np.clip(cy_delta_min, -th_y * size_min, th_y * size_min)
        cy_max = cy - np.clip(cy_delta_max, -th_y * size_max, th_y * size_max)
    
        self.bbox_min = np.hstack((
            cy_min - 0.5 * size_min,
            cx_min - 0.4 * size_min,
            cy_min + 0.5 * size_min,
            cx_min + 0.4 * size_min)).astype(np.int32)

        self.bbox_max = np.hstack((
            cy_max - 0.5 * size_max,
            cx_max - 0.5 * size_max,
            cy_max + 0.5 * size_max,
            cx_max + 0.5 * size_max)).astype(np.int32)


class PGGAN_published_100k(CelebA_HQ):

    def _set(self, kwargs):
        self.num_img = 100 * 1000


class PGGAN_published_nGPUs(Imdb):

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, self.name)

        imgpath = list()
        bbox_original = list()
        imgsize = list()
        for dirpath, dirnames, filenames in os.walk(root):
            if len(filenames) > 0 and len(dirnames) == 0:
                for fn in filenames:
                    if 'tick=' in fn:
                        p = re.compile('_tick=\d+_')
                        tick = int(p.search(fn).group()[6:-1])
                        # tick = int(fn.split('_')[-2].split('=')[-1])
                        resolution = 2 ** int((tick - 1 + 3000) / 1200)
                        bbox = [0, 0, resolution, resolution]
                        _dirpath = dirpath[len(self.cfg.data_path) + 1:]
                        imgpath.append(os.path.join(_dirpath, fn))
                        bbox_original.append(bbox)
                        imgsize.append([resolution, resolution])

        self.imgpath = np.array(imgpath)
        self.bbox_original = np.array(bbox_original, dtype=np.int32)
        self.imgsize = np.array(imgsize, dtype=np.int32)
    
    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)
        # ox_ratio = 212 / 1024
        # oy_ratio = 170 / 1024
        # minx_ratio = 600 / 1024
        # miny_ratio = 650 / 1024

        # for i in range(len(self.bbox_original)):
        #     h = self.bbox_original[i][2] - self.bbox_original[i][0]
        #     w = self.bbox_original[i][3] - self.bbox_original[i][1]
        #     if h >= 512 and w >= 512:
        #         top = int(h * ox_ratio)
        #         left = int(w * oy_ratio)
        #         height = int(h * minx_ratio)
        #         width = int(w * miny_ratio)
        #         self.bbox_min.append([top, left, top + height, left + width])
        #         self.bbox_max.append(self.bbox_original[i])
        #     else:
        #         self.bbox_min.append(self.bbox_original[i])
        #         self.bbox_max.append(self.bbox_original[i])

        # self.bbox_min = np.array(self.bbox_min, dtype=np.int32)
        # self.bbox_max = np.array(self.bbox_max, dtype=np.int32)


class PGGAN_trained(PGGAN_published_nGPUs):
    pass


class Glow(Imdb):

    def _set(self, kwargs):
        if 'test' in kwargs:
            self.test = kwargs['test']
        else:
            self.test = None

    def _parse_annotation(self):
        if self.test:
            imglist = glob(os.path.join(self.cfg.data_path, self.name, 'test', '*.jpg'))
        else:
            imglist = glob(os.path.join(self.cfg.data_path, self.name, 'train', '*.jpg'))

        imglist = ['/'.join(_.split('/')[-3:]) for _ in imglist]
        self.imgpath = np.array(imglist)
        self.bbox_original = np.array([[0, 0, 256, 256]] * len(imglist), dtype=np.int32)
        self.imgsize = np.array([[256, 256]] * len(imglist), dtype=np.int32)
    
    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)


class Glow_CelebA_HQ(Glow):
    pass


class Glow_UMDFaces(Glow):
    pass


class StarGAN_CelebA(Glow):
    pass


class BEGAN(Imdb):

    def _set(self, kwargs):
        raise NotImplementedError()

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, self.name)

        for dirpath, dirnames, filenames in os.walk(root):
            if len(filenames) > 0 and len(dirnames) == 0:
                for fn in filenames:
                    p = re.compile('_size=\d+_')
                    size = int(p.search(fn).group()[6:-1])
                    bbox = [0, 0, size, size]
                    _dirpath = dirpath[len(self.cfg.data_path) + 1:]
                    self.imgpath.append(os.path.join(_dirpath, fn))
                    self.bbox_original.append(bbox)
    
    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)


class DCGAN(Imdb):

    def _set(self, kwargs):
        raise NotImplementedError()
 
    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, self.name)

        for dirpath, dirnames, filenames in os.walk(root):
            if len(filenames) > 0 and len(dirnames) == 0:
                for fn in filenames:
                    bbox = [0, 0, 64, 64]
                    _dirpath = dirpath[len(self.cfg.data_path) + 1:]
                    self.imgpath.append(os.path.join(_dirpath, fn))
                    self.bbox_original.append(bbox)
        

    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)


class LSGAN(DCGAN):
    pass


class DRAGAN(DCGAN):
    pass


class WGAN_GP(DCGAN):
    pass


class Photoshop(Imdb):

    def __init__(self, cfg, name=None, virtual=False, logger=None, **kwargs):
        super(Photoshop, self).__init__(cfg, name=name, virtual=virtual, logger=logger, **kwargs)

        weight = 0
        for k, v in self.cfg.dataset.swap_parts.items():
            weight += v

        assert math.isclose(weight, 1.0), 'Sum of swap_parts weight should be 1.0, but {}'.format(weight)

    def _parse_annotation(self):
        with open(os.path.join(self.cfg.data_path, 'Photoshop', 'Photoshop_result.csv')) as f:
            imgpath = list()
            imgsize = list()
            bbox = list()
            partname = list()
            level = list()
            bbox_mask = list()

            rdr = csv.DictReader(f)
            for row in rdr:
                imgpath.append(os.path.join(
                    'Photoshop', 
                    'Photoshop_' + self.cfg.dataset.photoshop_ext, 
                    row['filename'] + '.' + self.cfg.dataset.photoshop_ext)
                )
                imgsize.append([int(row['h']), int(row['w'])])
                bbox.append([
                    int(row['face_y']),
                    int(row['face_x']),
                    int(row['face_y']) + int(row['face_h']),
                    int(row['face_x']) + int(row['face_w'])])
                partname.append(row['partname'])
                level.append(int(row['level']))
                bbox_mask.append([
                    int(row['mask_y']),
                    int(row['mask_x']),
                    int(row['mask_y']) + int(row['mask_h']),
                    int(row['mask_x']) + int(row['mask_w'])
                ])

        self.imgpath = np.array(imgpath)
        self.imgsize = np.array(imgsize, dtype=np.int32)
        self.bbox_original = np.array(bbox, dtype=np.int32)

        self.partname = np.array(partname)
        self.level = np.array(level, dtype=np.int32)
        self.bbox_mask = np.array(bbox_mask, dtype=np.int32)

    def _filter(self):
        super(Photoshop, self)._filter()

        partname_valid = [key for key, value in self.cfg.dataset.swap_parts.items() if value > 0]        
        mask_part = np.zeros(len(self.imgpath), dtype=np.bool)
        for pv in partname_valid:
            mask_part = mask_part | (self.partname == pv)
        
        mask = mask_part

        self.imgpath = self.imgpath[mask]
        self.imgsize = self.imgsize[mask]
        self.bbox_original = self.bbox_original[mask]
        self.bbox_min = self.bbox_min[mask]
        self.bbox_max = self.bbox_max[mask]
        self.minsize = self.minsize[mask]
        self.maxsize = self.maxsize[mask]
        if hasattr(self, 'angle'):
            self.angle = self.angle[mask]
        if hasattr(self, 'points'):
            self.points = self.points[mask]
        if hasattr(self, 'lab'):
            self.lab = self.lab[mask]
        if hasattr(self, 'partname'):
            self.partname = self.partname[mask]
        if hasattr(self, 'level'):
            self.level = self.level[mask]
        if hasattr(self, 'bbox_mask'):
            self.bbox_mask = self.bbox_mask[mask]

    def _set_bbox_range(self):
        top, left, bottom, right = np.split(self.bbox_original, np.arange(1, 4), axis=1)
        h = bottom - top
        w = right - left
        size_min = np.max((h, w), 0) * 1.1
        size_max = np.max((h, w), 0) * 1.5
        cx = np.mean((left, right), 0)
        cy = np.mean((top, bottom), 0) - 0.1 * h

        self.bbox_min = np.hstack((
            cy - 0.5 * size_min,
            cx - 0.4 * size_min,
            cy + 0.5 * size_min,
            cx + 0.4 * size_min)).astype(np.int32)

        self.bbox_max = np.hstack((
            cy - 0.5 * size_max,
            cx - 0.5 * size_max,
            cy + 0.5 * size_max,
            cx + 0.5 * size_max)).astype(np.int32)


class Sample_1_M1_Real(Imdb):
    ''' 201~400 real images from Sample_1, task 1 '''

    def _parse_annotation(self):

        with open(os.path.join(self.cfg.data_path, 'RnDChallenge', 'Sample_1_M1_imsize.csv'), 'r') as f:
            rdr = csv.DictReader(f)

            imgpath = list()
            bbox_original = list()
            imgsize = list()
            for row in rdr:
                idx = int(re.findall('\d{5}', row['filename'])[0])
                if idx > 200:
                    imgpath.append(os.path.join('RnDChallenge', 'Sample_1', row['filename']))
                    h = int(row['h'])
                    w = int(row['w'])
                    bbox_original.append([0, 0, h, w])
                    imgsize.append([h, w])
            
            self.imgpath = np.array(imgpath)
            self.bbox_original = np.array(bbox_original, dtype=np.int32)
            self.imgsize = np.array(imgsize, dtype=np.int32)

    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)


class Sample_1_M2_Real(Imdb):
    ''' Real images from Sample_1, task 2 '''

    def _set(self, kwargs):
        self.ignore_cache = True
        self.real_fake = 'real'
        self.subdir = 'Sample_1'
        self.anno_file = 'Sample_1_M2_bbox.csv'

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, 'RnDChallenge')

        with open(os.path.join(root, self.anno_file)) as f:
            imgpath = list()
            imgsize = list()
            bbox = list()

            rdr = csv.DictReader(f)
            for row in rdr:
                if row['real_fake'] != self.real_fake:
                    continue

                imgpath.append(os.path.join('RnDChallenge', self.subdir, row['filename']))
                imgsize.append([int(row['h']), int(row['w'])])
                bbox.append([
                    int(row['face_y']),
                    int(row['face_x']),
                    int(row['face_y']) + int(row['face_h']),
                    int(row['face_x']) + int(row['face_w'])])

        self.imgpath = np.array(imgpath)
        self.imgsize = np.array(imgsize, dtype=np.int32)
        self.bbox_original = np.array(bbox, dtype=np.int32) 

    def _set_bbox_range(self):
        top, left, bottom, right = np.split(self.bbox_original, np.arange(1, 4), axis=1)
        h = bottom - top
        w = right - left
        size_min = np.max((h, w), 0) * 1.1
        size_max = np.max((h, w), 0) * 1.5
        cx = np.mean((left, right), 0)
        cy = np.mean((top, bottom), 0) - 0.1 * h

        self.bbox_min = np.hstack((
            cy - 0.5 * size_min,
            cx - 0.4 * size_min,
            cy + 0.5 * size_min,
            cx + 0.4 * size_min)).astype(np.int32)

        self.bbox_max = np.hstack((
            cy - 0.5 * size_max,
            cx - 0.5 * size_max,
            cy + 0.5 * size_max,
            cx + 0.5 * size_max)).astype(np.int32)


class Sample_1_M2_Syn(Sample_1_M2_Real):
    ''' Fake images from Sample_1, task 2 '''

    def _set(self, kwargs):
        self.ignore_cache = True
        self.real_fake = 'fake'
        self.subdir='Sample_1'
        self.anno_file = 'Sample_1_M2_bbox.csv'


class Sample_1_M1_GAN(Imdb):
    ''' 1~200 fake images from Sample_1, task 1 '''

    def _parse_annotation(self):

        with open(os.path.join(self.cfg.data_path, 'RnDChallenge', 'Sample_1_M1_imsize.csv'), 'r') as f:
            rdr = csv.DictReader(f)

            imgpath = list()
            bbox_original = list()
            imgsize = list()
            for row in rdr:
                idx = int(re.findall('\d{5}', row['filename'])[0])
                if idx <= 200:
                    imgpath.append(os.path.join('RnDChallenge', 'Sample_1', row['filename']))
                    h = int(row['h'])
                    w = int(row['w'])
                    bbox_original.append([0, 0, h, w])
                    imgsize.append([h, w])
            
            self.imgpath = np.array(imgpath)
            self.bbox_original = np.array(bbox_original, dtype=np.int32)
            self.imgsize = np.array(imgsize, dtype=np.int32)

    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)


class Sample_2_GAN(Imdb):
    ''' 60 gan images from Sample_2 '''

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, 'RnDChallenge', 'Sample_2', 'gan_jpg')

        imglist = glob(os.path.join(root, 'gan_*.jpg'))
        imgpath = [os.path.join('RnDChallenge', 'Sample_2', 'gan_jpg', os.path.basename(_)) for _ in imglist]
        bbox_original = [[0, 0, 128, 128]] * len(imgpath)
        imgsize = [[0, 0, 128, 128]] * len(imgpath)

        assert len(imgpath) > 0

        self.imgpath = np.array(imgpath)
        self.bbox_original = np.array(bbox_original, dtype=np.int32)
        self.imgsize = np.ones((self.imgpath.shape[0], 2), dtype=np.int32) * 128

    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)


class Sample_2_Syn(Imdb):
    ''' Fake images from Sample_2 '''

    def _set(self, kwargs):
        self.ignore_cache = True

    def _parse_annotation(self):
        root = os.path.join(self.cfg.data_path, 'RnDChallenge', 'Sample_2', 'syn')

        imgpath = glob(os.path.join(root, '*.jpg'))
        imgpath = ['/'.join(_.split('/')[-4:]) for _ in imgpath]
        bbox_original = [[0, 0, 128, 128]] * len(imgpath)

        assert len(imgpath) > 0

        self.imgpath = np.array(imgpath)
        self.bbox_original = np.array(bbox_original, dtype=np.int32) 
        self.imgsize = np.ones((self.imgpath.shape[0], 2), dtype=np.int32) * 128

    def _set_bbox_range(self):
        self.bbox_min = np.copy(self.bbox_original)
        self.bbox_max = np.copy(self.bbox_original)
