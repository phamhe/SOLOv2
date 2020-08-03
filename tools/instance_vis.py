import os
import random

from pycocotools.coco import COCO
import cv2
from matplotlib import pyplot as plt

json_file = '/hexiao/dataset/Fashionpedia/instances_attributes_train2020_clean_v2.json'
dataset_dir = '/hexiao/dataset/Fashionpedia/train/'
dst_dir = '/hexiao/cache_dir/deepfashion2_garment_gt'
coco = COCO(json_file)
catIds = coco.getCatIds()
imgIds = coco.getImgIds()
#imgIds = coco.getImgIds(imgIds=imgIds, catIds=catIds)
#idxs = random.sample(range(0, 44776), 100)
idxs = range(0, 100)
for i in idxs:
    print("processing:", i) 
    img = coco.loadImgs(imgIds[i])[0]
    I = cv2.imread(dataset_dir + img['file_name'])
    plt.axis('off')
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.savefig(os.path.join(dst_dir, '%s.jpg'%i))
    plt.clf()
    

