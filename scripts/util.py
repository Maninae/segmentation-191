from os.path import join

coco_dir = "../coco"
anno_dir = join(coco_dir, "annotations")
img_dir  = join(coco_dir, "images")
tv_anno_dir = join(anno_dir, "annotations_trainval2017")

val_anno_file = join(tv_anno_dir, "instances_val2017.json")
train_anno_file = join(tv_anno_dir, "instances_train2017.json")

train_img_dir = join(img_dir, "people-train2017")
val_img_dir   = join(img_dir, "people-val2017")
train_mask_dir = join(img_dir, "mask-train2017")
val_mask_dir   = join(img_dir, "mask-val2017")

def wrapper(d):
    return d + "-wrapper"

train_img_dir_wrapper = wrapper(train_img_dir)
val_img_dir_wrapper = wrapper(val_img_dir)
train_mask_dir_wrapper = wrapper(train_mask_dir)
val_mask_dir_wrapper = wrapper(val_mask_dir)



def cocoTrain():
    from pycocotools.coco import COCO
    return COCO(train_anno_file)

def cocoVal():
    from pycocotools.coco import COCO
    return COCO(val_anno_file)
