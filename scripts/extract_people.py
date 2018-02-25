from os.path import join
from os import rename
from pycocotools.coco import COCO
from sys import exit
from PIL import Image
import numpy as np
from pycocotools.mask import merge

coco_dir = "../coco"
anno_dir = join(coco_dir, "annotations/annotations_trainval2017")
img_dir_val  = join(coco_dir, "images/coco-val2017")
img_dir_train = join(coco_dir, "images/coco-train2017")
out_dir_train  = join(coco_dir, "images/people-train2017")
out_dir_val = join(coco_dir, "images/people-val2017")

img_dir = join(coco_dir, "images")
mask_dir_train = join(img_dir, "mask-train2017-wrapper/mask-train2017")
mask_dir_val = join(img_dir, "mask-val2017-wrapper/mask-val2017")

val_annFile = join(anno_dir, "instances_val2017.json")
cocoVal = COCO(val_annFile)
train_annFile = join(anno_dir, "instances_train2017.json")
cocoTrain = COCO(train_annFile)

person_cat_id = cocoTrain.getCatIds(catNms=['person'])

def get_person_img_ids():
    trainCatIds = cocoTrain.getCatIds(catNms=['person'])
    trainImgIds = cocoTrain.getImgIds(catIds=trainCatIds)

    valCatIds = cocoVal.getCatIds(catNms=['person'])
    valImgIds = cocoVal.getImgIds(catIds=valCatIds)

    print("Sample of train img ids: %s" % str(trainImgIds[:5]))
    print("Sample of val img ids:   %s" % str(valImgIds[:5]))

    return trainImgIds, valImgIds

def as_filenames(imgids):
    return ['%012d.jpg' % i for i in imgids]
	
def transfer_files(filenames, source_dir, dest_dir):
    counter = 0
    for filename in filenames:
        rename(join(source_dir, filename), join(dest_dir, filename))
        counter += 1
        if counter % 1000 == 0:
            print("At file %d ..." % counter)
    print("Transferred %d files from %s to %s." % (counter, source_dir, dest_dir))


def generate_masks(coco, img_ids, out_dir):

    def union_of_masks(masks):
        mask = np.zeros(masks[0].shape)
        for m in masks:
            mask += m
        mask[mask > 1] = 1
        return mask

    print("Generating masks for out_dir: %s" % out_dir)
    print("Sample image_ids: %s" % str(img_ids[:5]))

    annIds = coco.getAnnIds(imgIds=img_ids, catIds=person_cat_id, iscrowd=None)
    print("Retrieved annIds: %s..." % str(annIds[:5]))
    anns = coco.loadAnns(annIds)
    print("Retrieved annotations. There are %d." % len(anns))

    counter = 0
    all_anns = {}
    for ann in anns:
        image_id = ann['image_id']
        filename = '%012d.png' % image_id
        
        if filename not in all_anns:
            all_anns[filename] = [ann]
        else:
            all_anns[filename].append(ann)
            print("We found two anns under %s. appended, counter %d." % (filename, counter))
        counter += 1

    counter = 0
    print("Saving the anns as masks now.")
    for filename in all_anns:
        anns = all_anns[filename]
        masks = [coco.annToMask(ann) for ann in anns]
        
        mask = union_of_masks(masks)

        maskimg = Image.fromarray(np.uint8(mask * 255), 'L')
        maskimg.save(join(out_dir, filename), 'PNG')
        counter += 1

        if counter % 1000 == 0:
            print("Processed %d... saved mask %s." % (counter, filename))



if __name__ == "__main__":
    tii, vii = get_person_img_ids()
    # tfn, vfn = as_filenames(tii), as_filenames(vii)
    # transfer_files(tfn, img_dir_train, out_dir_train)
    # transfer_files(vfn, img_dir_val, out_dir_val)

    train_image_ids, val_image_ids = tii, vii

    generate_masks(cocoTrain, train_image_ids, mask_dir_train)
    generate_masks(cocoVal, val_image_ids, mask_dir_val)
