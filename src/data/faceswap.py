import cv2
import numpy as np
import random

def applyAffineTransform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=random.randrange(1,5), borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    pt = []    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []        
    
    return delaunayTri, subdiv

def warpTriangle(img1, img2, t1, t2) :
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    size = (r2[2], r2[3])
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 

def cvt_annot_to_pt(annot):

    pt_list = []
    for x, y in annot:
        pt_list.append((int(x), int(y)))
    return pt_list

def filter_keyid_annot(src_annot, filter_ids):
    """ src_annot: a list of (keypoint_id, x, y)
        filter_ids: a list of [keypoint_ids] to be remained
    """
    new_list = []
    for item in src_annot:
        if item[0] in filter_ids:
            new_list.append(item)
    
    return new_list

def draw_face_landmark(img, annot):

    for x, y in annot:
        cv2.circle(img, (int(x),int(y)), 3, (0,0,255), -1)

def calc_new_keypoints(annot):
    """ Caculate additional 11 keypoints from existing 21 keypoints.
        For visual representation of new key points, please refer to the reference report.
        annot: 
    """
    # assert len(annot) == 21, "To add new key points, existing annotation should be 21, but {}.".format(len(annot))

    b1_annot = []
    b1_annot.append(np.zeros((2,), np.int32))   # add dummy array to make annot list to 1 base index.
    b1_annot.extend(annot)

    # below eyeblows
    i22 = (b1_annot[2] + b1_annot[8])/2     #left
    b1_annot.append(i22)
    i23 = (b1_annot[5] + b1_annot[11])/2    #right
    b1_annot.append(i23)

    # temples
    i24 = b1_annot[7] + (b1_annot[7] - b1_annot[9])/2   #left
    b1_annot.append(i24)
    i25 = b1_annot[12]+ (b1_annot[12] - b1_annot[10])/2     #right
    b1_annot.append(i25)

    # the ridge of the nose
    i26 = (b1_annot[9] + b1_annot[10] + b1_annot[15])/3
    b1_annot.append(i26)

    #cheeks
    i27 = (b1_annot[13] + b1_annot[14])/2     #left
    b1_annot.append(i27)
    i28 = (b1_annot[16] + b1_annot[17])/2    #right
    b1_annot.append(i28)

    #philtrum
    i29 = (b1_annot[15] + b1_annot[19])/2
    b1_annot.append(i29)

    #upper chin
    i30 = (b1_annot[19] + b1_annot[21])/2
    b1_annot.append(i30)

    #below eyes
    i31 = (b1_annot[7] * 2 + b1_annot[15])/3     #left
    b1_annot.append(i31)
    i32 = (b1_annot[12] * 2 + b1_annot[15])/3    #right
    b1_annot.append(i32)

    #cheek bones
    i33 = (b1_annot[27] + b1_annot[31])/2   #left
    b1_annot.append(i33)
    i34 = (b1_annot[28] + b1_annot[32])/2   #right
    b1_annot.append(i34)

    #below nose
    i35 = (b1_annot[14] + b1_annot[18])/2   #left
    b1_annot.append(i35)
    i36 = (b1_annot[16] + b1_annot[20])/2   #right
    b1_annot.append(i36)

    #below lips
    r1 = (b1_annot[18] + b1_annot[21])/2    #left reference
    r2 = (b1_annot[20] + b1_annot[21])/2    #right reference
    i37 = (r1+r2)/2 + 3*(r1-r2)/4           #left
    b1_annot.append(i37)
    i38 = (r1+r2)/2 + 3*(r2-r1)/4           #right
    b1_annot.append(i38)

    #sides of lips
    i39 = b1_annot[18] + (b1_annot[18] - b1_annot[19])/3
    b1_annot.append(i39)
    i40 = b1_annot[20] + (b1_annot[20] - b1_annot[19])/3
    b1_annot.append(i40)

    assert len(b1_annot) - 1 == 40, "the length of new annot list should be 32 but {}".format(len(b1_annot) - 1)

    return b1_annot[1:]

def getBoundaryMask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    new_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    return new_mask

def faceswap(img1, face_rect1, annot1, img2, face_rect2, annot2, style, rect_crop, resolution):
    """ crop img1's face and merge into img2's face region
        face1 and face2 have the identical keypoints.
    """
    img1Warped = np.copy(img2);    

    # Read array of corresponding points
    points1 = cvt_annot_to_pt(annot1)
    points2 = cvt_annot_to_pt(annot2)
    
    # # Find convex hull
    hull1 = []
    hull2 = []
    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
    for i in range(len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
    # # print('hull1: {}, hull2: {}'.format(hull1, hull2))
    # # Find delanauy traingulation for convex hull points
    # dt, subdiv = calculateDelaunayTriangles(face_rect2, hull2)

    # if len(dt) == 0:
    #     raise Exception()

    # # Apply affine transformation to Delaunay triangles
    # for i in range(len(dt)):
    #     t1, t2 = [], []
    #     #get points for img1, img2 corresponding to the triangles
    #     for j in range(3):
    #         t1.append(hull1[dt[i][j]])
    #         t2.append(hull2[dt[i][j]])

    #     warpTriangle(img1, img1Warped, t1, t2)

    dt, subdiv = calculateDelaunayTriangles(face_rect2, points2)
    # Apply affine transformation to Delaunay triangles
    for i in range(len(dt)):
        t1, t2 = [], []
        #get points for img1, img2 corresponding to the triangles
        for j in range(3):
            t1.append(points1[dt[i][j]])
            t2.append(points2[dt[i][j]])

        warpTriangle(img1, img1Warped, t1, t2)

    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    
    mask = np.zeros(img2.shape, dtype = img2.dtype)
    # img_h, img_w = img2.shape[0:2]
    # mask_1ch = np.zeros((img_h, img_w, 1), dtype = np.uint8)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    # cv2.fillConvexPoly(mask_1ch, np.int32(hull8U), (255, 255, 255))

    # x, y, w, h = face_rect2
    # cx = int(x + 0.5 * w)
    # cy = int(y + 0.5 * h)
    # maxwh = max(w, h)
    # half = int(0.5 * (1 + 2 * arg.pad) * maxwh)
    # x = max(0, cx - half)
    # y = max(0, cy - half)
    # w = 2 * half
    # h = 2 * half

    # img1Warped = img1Warped[y:y+h, x:x+w, :]
    # img1Warped = cv2.resize(img1Warped, (arg.resize, arg.resize), cv2.INTER_CUBIC)
    top, left, bottom, right = rect_crop
    mask = mask[top:bottom, left:right, :]
    mask = cv2.resize(mask, (resolution, resolution), cv2.INTER_NEAREST)
    img1Warped = img1Warped[top:bottom, left:right, :]
    img1Warped = cv2.resize(img1Warped, (resolution, resolution), cv2.INTER_CUBIC)
    resize_ratio_x = resolution / (right - left)
    resize_ratio_y = resolution / (bottom - top)

    if style == 'warp':
        # output = img1Warped
        output = cv2.colorChange(np.uint8(img1Warped), mask, random.uniform(0.5, 2.5), random.uniform(0.5, 2.5), random.uniform(0.5, 2.5))
    elif style == 'smoothwarp':
        output = cv2.colorChange(np.uint8(img1Warped), mask, random.uniform(0.5, 2.5), random.uniform(0.5, 2.5), random.uniform(0.5, 2.5))
        output_blur = cv2.GaussianBlur(output, (7,7), random.uniform(0.8, 2.0))
        b_mask = getBoundaryMask(mask)
        b_mask = b_mask.astype(np.bool)
        output[b_mask] = output_blur[b_mask]
        
    elif style == 'seamless':
        hx, hy, hw, hh = cv2.boundingRect(np.float32([hull2]))
        hull_top = max(top, hy)
        hull_left = max(left, hx)
        hull_bottom = min(bottom, hy + hh)
        hull_right = min(right, hx + hw)
        center_hull = (int((hull_left + hull_right) / 2), int((hull_top + hull_bottom) / 2))
        # center_hull = (int(hx + hw / 2), int(hy + hh / 2))

        img2 = img2[top:bottom, left:right, :]
        img2 = cv2.resize(img2, (resolution, resolution), cv2.INTER_CUBIC)

        chx, chy = center_hull
        chx = int((chx - left) * resize_ratio_x)
        chy = int((chy - top) * resize_ratio_y)
        center = (chx, chy)

        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    # for debugging - KJH
    # temp_annot=[]
    # for x,y in annot2:
    #     r_x = int((x - left) * resize_ratio_x)
    #     r_y = int((y - top) * resize_ratio_y)
    #     temp_annot.append((r_x,r_y))
    # draw_face_landmark(output, temp_annot)

    # mask = getBoundaryMask(mask)
        
    return output, mask

if __name__ == '__main__':
    import sys; sys.path.append('../')
    from imdb import UMDFaces
    from dataset import DatasetSyn
    from util import get_config, Logger
    import datetime
    import argparse
    import os
    from glob import glob
    from tqdm import trange

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--tag', default="faceswap", type=str, help='Experiment tag')
    parser.add_argument('--name', type=str, default='UMDFaces')
    parser.add_argument('--num_img', type=int, default=300)
    parser.add_argument('--clear_cache', action='store_true')
    parser.add_argument('--preset', type=str, default='gan')
    parser.add_argument('--set', type=str, default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_config(args, now, src_dir=os.path.join(os.getcwd(), '../'))

    print('>> Loading Logger...')
    logger = Logger(cfg, None)

    class_name = args.name.replace('-', '_')
    imdb = eval(class_name)(cfg, args.name, logger=logger)
    dset = DatasetSyn(cfg, imdb, 'train', 'fake')
    # dset.swap_style = ['seamless']
    dset.swap_style = { 'warp': 0.0, 'smoothwarp': 1.0, 'seamless': 0.0 }

    check_path = os.path.join(cfg.test_path, 'check_faceswap', args.name)
    os.makedirs(check_path, exist_ok=True)

    import cv2
    li = list(range(len(imdb.imgpath)))
    import random
    # print('>> Shuffling...')
    # random.shuffle(li)

    print('>> Removing existing files...')
    for filename in glob(os.path.join(check_path, '*.jpg')):
        os.remove(filename)

    dset.reset()

    print('>> Start!')
    for i in trange(args.num_img):
        idx = li[i]
        idx_src = idx
        idx_dst = dset._get_idx_dst(idx_src)
        crop_rect = dset._crop(idx_dst)

        imgpath = os.path.join(cfg.data_path, imdb.imgpath[idx_dst])
        bbox_original = imdb.bbox_original[idx]
        bbox_min = imdb.bbox_min[idx]
        bbox_max = imdb.bbox_max[idx]
        img = cv2.imread(imgpath)
        h, w, c = img.shape
        # thickness = max(5, 5 * h // 500)
        thickness = 1

        imgpath_src = os.path.join(cfg.data_path, imdb.imgpath[idx_src])
        img_src = cv2.imread(imgpath_src)
        ann_src = imdb.points[idx_src]
        new_ann_src = calc_new_keypoints(ann_src)
        draw_face_landmark(img_src, new_ann_src)
        # img = cv2.rectangle(
        #     img, 
        #     (bbox_original[1], bbox_original[0]), 
        #     (bbox_original[3], bbox_original[2]), 
        #     (0, 0, 255), thickness) # Red

        # img = cv2.rectangle(
        #     img, 
        #     (bbox_min[1], bbox_min[0]), 
        #     (bbox_min[3], bbox_min[2]), 
        #     (255, 0, 0), thickness) # Blue

        # img = cv2.rectangle(
        #     img, 
        #     (bbox_max[1], bbox_max[0]), 
        #     (bbox_max[3], bbox_max[2]), 
        #     (255, 0, 0), thickness) # Blue
        
        img = cv2.rectangle(
            img,
            (crop_rect[1], crop_rect[0]),
            (crop_rect[3], crop_rect[2]),
            (0, 0, 255), thickness) # Red

        ann_dst = imdb.points[idx_dst]
        new_ann_dst = calc_new_keypoints(ann_dst)
        draw_face_landmark(img, new_ann_dst)

        imgpath = imgpath.replace(cfg.data_path + '/', '')
        imgpath = imgpath.replace(args.name + '/', '')
        imgpath = imgpath.replace('.jpg', '')
        imgpath = imgpath.replace('.png', '')
        imgname = imgpath.replace('/', '+')

        save_img_name = os.path.join(check_path, '{:03d}_{}_img.jpg'.format(i, imgname))
        save_face_name = os.path.join(check_path, '{:03d}_{}_face.jpg'.format(i, imgname))
        save_mask_name = os.path.join(check_path, '{:03d}_{}_mask.jpg'.format(i, imgname))

        # cv2.imwrite(save_img_name, img)
        
        save_img_name_src = save_img_name.replace('img.jpg', 'srcimg.jpg')
        #cv2.imwrite(save_img_name_src, img_src)

        # face, mask = dset._random_swap(idx_src, idx_dst, crop_rect)
        face = dset._random_swap(idx_src, idx_dst, crop_rect)

        # cv2.imwrite(save_face_name, face)
        # cv2.imwrite(save_mask_name, mask)


