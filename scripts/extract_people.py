from os.path import join
from os import rename
from pycocotools.coco import COCO

coco_dir = "../coco"
anno_dir = join(coco_dir, "annotations/annotations_trainval2017")
img_dir_val  = join(coco_dir, "images/coco-val2017")
img_dir_train = join(coco_dir, "images/coco-train2017")
out_dir  = "../tmp"

val_annFile = join(anno_dir, "instances_val2017.json")
cocoVal = COCO(val_annFile)
train_annFile = join(anno_dir, "instances_train2017.json")
cocoTrain = COCO(train_annFile)

def get_person_img_ids():
	trainCatIds = cocoTrain.getCatIds(catNms=['person'])
	trainImgIds = cocoTrain.getImgIds(catIds=catIds)

	valCatIds = cocoVal.getCatIds(catNms=['person'])
    valImgIds = cocoVal.getImgIds(catIds=catIds)

    print("Sample of train img ids: %s" % str(trainImgIds[:5]))
    print("Sample of val img ids:   %s" % str(valImgIds[:5]))

    return trainImgIds, valImgIds

def as_filenames(tii, vii):
	pass
	
if __name__ == "__main__":
	tii, vii = get_person_img_ids()