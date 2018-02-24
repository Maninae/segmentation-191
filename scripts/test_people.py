from os.path import join
from os import rename, listdir
from pycocotools.coco import COCO
from sys import exit
from numpy.random import randint

coco_dir = "../coco"
anno_dir = join(coco_dir, "annotations/annotations_trainval2017")
# img_dir_val  = join(coco_dir, "images/coco-val2017")
# img_dir_train = join(coco_dir, "images/coco-train2017")
out_dir_train  = join(coco_dir, "images/people-train2017")
out_dir_val = join(coco_dir, "images/people-val2017")

val_annFile = join(anno_dir, "instances_val2017.json")
cocoVal = COCO(val_annFile)
train_annFile = join(anno_dir, "instances_train2017.json")
cocoTrain = COCO(train_annFile)

def get_person_img_ids():
    trainCatIds = cocoTrain.getCatIds(catNms=['person'])
    trainImgIds = cocoTrain.getImgIds(catIds=trainCatIds)

    valCatIds = cocoVal.getCatIds(catNms=['person'])
    valImgIds = cocoVal.getImgIds(catIds=valCatIds)

    print("Sample of train img ids: %s" % str(trainImgIds[:5]))
    print("Sample of val img ids:   %s" % str(valImgIds[:5]))

    return trainImgIds, valImgIds


def test_images_loading(coco, xii, desired_dir_files):
    # xii := X image Ids. Train/val image ids.
    print("Testing loading the images.")

    images = coco.loadImgs(xii)
    n_images = len(images)
    print("We loaded %d images." % n_images)

    i = 0
    for image in images:
        assert(image['file_name'] in desired_dir_files)

        if i % 1000  == 0:
            print("At test %d..." % i)
        i += 1

if __name__ == "__main__":
    tii, vii = get_person_img_ids()

    print("Testing VAL.")
    val_files = set([f for f in listdir(out_dir_val) if f[-4:] == '.jpg'])
    test_images_loading(cocoVal, vii, val_files)

    print("Testing TRAIN.")
    train_files = set([f for f in listdir(out_dir_train) if f[-4:] == '.jpg'])
    test_images_loading(cocoTrain, tii, train_files)
