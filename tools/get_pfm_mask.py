#!/usr/bin/env python
# encoding: utf-8
################################################################
# File Name: tools/get_pfm_mask.py
# Author: gaoyu
# Mail: 1400012705@pku.edu.cn
# Created Time: 2020-07-03 09:39:51
################################################################
import os
import re
import numpy as np
import uuid
from scipy import misc
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import cv2
import pickle

def readPFM(file):
    file = open(file, 'rb')
 
    color = None
    width = None
    height = None
    scale = None
    endian = None
     
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
        
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')
     
    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
         
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
     
    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale
         
def writePFM(file, image, scale=1):

    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
     
    image = np.flipud(image)
     
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
        
    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
     
    endian = image.dtype.byteorder
     
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
        file.write('%f\n'.encode() % scale)
        image.tofile(file)


if __name__ == '__main__':

    # init PFM
    #src_dir = '/hexiao/dataset/20200622_grey'
    #dst_dir = '/hexiao/dataset/20200622_pfm'
    #imgs = [img for img in os.listdir(src_dir) if '.jpg' in img]

    #for img in imgs:
    #    pfm_img = os.path.join(dst_dir, '%s.pfm'%(img.strip('.jpg')))
    #    im = cv2.imread(os.path.join(src_dir, img), cv2.IMREAD_GRAYSCALE)
    #    im = np.array(im, dtype=np.float32)
    #    writePFM(pfm_img, im)

    src_pfm_dir = '/hexiao/dataset/20200622_pfm'
    res_path = '/hexiao/dataset/0702_test_output/output.pkl'
    dst_dir = '/hexiao/dataset/20200622_pfm_mask'

    res = pickle.load(open(res_path, 'rb'))
    res_dic = {}
    for frame in res:
        if os.path.basename(frame['filename']) not in res_dic:
            res_dic[os.path.basename(frame['filename'])] = {
                'mask':frame['mask']
            }
    imgs = [img for img in os.listdir(src_pfm_dir) if 'pfm' in img]
    for img in imgs:
        im, im_scale = readPFM(os.path.join(src_pfm_dir, img))
        mask = res_dic['%s.jpg'%(img.strip('.pfm'))]['mask']
        im = cv2.resize(im, (mask.shape[1], mask.shape[0]))
        im[mask] = 0
        cv2.imwrite(os.path.join(dst_dir, '%s.jpg'%(img.strip('.pfm'))), im)
        writePFM(os.path.join(dst_dir, img), im)


