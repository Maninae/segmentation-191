import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from os.path import join
from collections import Counter

from PIL import Image
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


coco_dir = "../coco"
anno_dir = join(coco_dir, "annotations/annotations_trainval2017")
img_dir  = join(coco_dir, "images/coco-val2017")


val_annFile = join(anno_dir, "instances_val2017.json")
coco = COCO(val_annFile)

catIds = coco.getCatIds(catNms=['person'])

imgIds = coco.getImgIds(catIds=catIds)
print("We found %d imgIds under the catIds %s ('person')" % (len(imgIds), str(catIds)))

print("Selecting an arbitrary image.")
img = coco.loadImgs(imgIds[0])
# This is a list so we access one more time.
img = img[0]

img_filename = img['file_name']
I = io.imread('%s/%s' % (img_dir, img_filename)) # This is an ndarray
plt.axis('off')
plt.imshow(I)
# plt.show()
print("Saving the image of a person.")
plt.imsave('person_img.png', I)

plt.imshow(I)
plt.axis('off')
# Only loads the annotations for that image, that has cats belongs under catIds
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) 
anns = coco.loadAnns(annIds)
# Sort of like matplotlib show
# coco.showAnns(anns)

ann = anns[0] # Just take one annotation
mask = coco.annToMask(ann) # Turn it into a binary mask for segmentation
print(mask.shape)
print(mask)
print("max: %f" % np.amax(mask))
print("min: %f" % np.amin(mask))
print("values are: %s" % str(Counter(mask.flatten())))

print("Saving the mask as a numpy array.")
np.save("person_mask.npy", mask)
print("Saving the mask as an image too.")
maskimg = Image.fromarray(np.uint8(mask * 255), 'L')
maskimg.save('mask.png', 'PNG') 
