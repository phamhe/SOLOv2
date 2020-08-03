import os
import numpy as np

from pycocotools.coco import COCO
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':

    #json_file = '/hexiao/dataset/Fashionpedia/instances_attributes_train2020_clean.json'
    #dataset_dir = '/hexiao/dataset/Fashionpedia/train/'
    #dst_dir = '/hexiao/cache_dir/fashion_pedia'
    json_file = '/hexiao/dataset/DeepFashion2/annotations/deepfashion2_train_10w.coco.json'
    dataset_dir = '/hexiao/dataset/DeepFashion2/train/image'
    dst_dir = '/hexiao/cache_dir/deep_fashion_origin'

    coco = COCO(json_file)
    
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()

    cats_origin = coco.cats
    new_cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape']
    #cats = new_cats
    cats = cats_origin
    new_catIds = []
    #for id in catIds:
    #    print(cats_origin[id]['name'], cats_origin[id]['supercategory'])
    #exit()
    for id in catIds:
        if cats_origin[id]['name'] in new_cats:
            new_catIds.append(id)
            print('exists : ', cats_origin[id]['name'])
        #elif 'body' in cats_origin[id]['supercategory']:
        elif 'clothes' in cats_origin[id]['supercategory']:
            print('not exists : ', cats_origin[id]['name'])
    #catIds = new_catIds
    cats_list = []
    for i in range(len(imgIds)):
        print("processing:", i) 
        out_name = ''
        im_write = False
        img = coco.loadImgs(imgIds[i])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            cat_id= ann['category_id']
            cat_name = cats_origin[cat_id]['name']
            out_name += '%s#'%(cat_name)
            #if cat_name not in cats_list and 'body' in cats_origin[cat_id]['supercategory']:
            if cat_name not in cats_list and 'clothes' in cats_origin[cat_id]['supercategory']:
                im_write = True
                cats_list.append(cat_name)
        if not im_write:
            continue
        I = cv2.imread(os.path.join(dataset_dir, img['file_name']))
        #print(os.path.join(dataset_dir, img['file_name']))
        plt.axis('off')
        plt.imshow(I)
        coco.showAnns(anns)
        plt.savefig(os.path.join(dst_dir, '%s.jpg'%out_name))
        plt.clf()
