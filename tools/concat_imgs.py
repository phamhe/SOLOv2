import cv2
import numpy as np
import os

src_dir_list = [
                '/hexiao/cache_dir/deepfashion2_val_gt',
                '/hexiao/cache_dir/deepfashion2_5w',
                '/hexiao/cache_dir/deepfashion2_10w',
                '/hexiao/cache_dir/fashionpedia5w'
                ]
source = [
          'deepfashion2_val_gt',
          'deepfashion2_5w',
          'deepfashion2_10w',
          'fashionpedia5w'
          ]
dst_dir = '/hexiao/cache_dir/compare'

for i in range(100):
    ims = []
    for idx, src_dir in enumerate(src_dir_list):
        im = cv2.imread(os.path.join(src_dir, '%s.jpg'%i))
        if source[idx] == 'deepfashion2_val_gt':
            im = im[60:420, 100:480]
            im=cv2.resize(im, (480, 640))
        ims.append(im)
    im_new = np.zeros((2560, 1920*4, 3))
    for idx, im in enumerate(ims):
        cv2.putText(im, source[idx], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        im_rz = cv2.resize(im, (1920, 2560))
        im_new[0:2560, (idx*1920):((idx+1)*1920), :] = im_rz
    cv2.imwrite(os.path.join(dst_dir, '%s.jpg'%i), im_new)
        
