

# Used in the main clause to specify one test
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
img_dir_train = join(coco_dir, "images/coco-train2017")
out_dir  = "../tmp"

# val 2017 is 5K images, so easy to explore
val_annFile = join(anno_dir, "instances_val2017.json")
coco = COCO(val_annFile)
train_annFile = join(anno_dir, "instances_train2017.json")
cocoTrain = COCO(train_annFile)

def statistics_on_persons():
    """
    Count some stats on the images with people in them.
    """
    catIds = cocoTrain.getCatIds(catNms=['person'])
    imgIds = cocoTrain.getImgIds(catIds=catIds)
    print("There are %d images under 'person' in the train 2017 split." % len(imgIds))

    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    print("There are %d iamges under 'person' in the val 2017 split." % len(imgIds))

    

def image_sizes():
    # This is NOT comprehensive. We are just sampling image sizes from val 2017, person category.
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    print("We found %d imgIds under the catIds %s ('person')" % (len(imgIds), str(catIds)))

    image_dims = [(img['height'], img['width']) for img in coco.loadImgs(imgIds)] 
    # coco.loadImgs(...) = A list of img's, where img = a 1-list of a dictionary.
    """
    {
        'date_captured': '2013-11-15 03:04:30', 
        'flickr_url': 'http://farm4.staticflickr.com/3117/2764199263_08af9e70bc_z.jpg', 
        'id': 507473, 
        'height': 480, 
        'license': 3, 
        'width': 640, 
        'file_name': '000000507473.jpg', 
        'coco_url': 'http://images.cocodataset.org/val2017/000000507473.jpg'
    }
    """
    counted_dims = Counter(image_dims)
    print("The dimensions of the 'person' images under val 2017, counted:")
    print(counted_dims)


def basic_demo():
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    print("We found %d imgIds under the catIds %s ('person')" % (len(imgIds), str(catIds)))

    print("Selecting an arbitrary image.")
    img = coco.loadImgs(imgIds[0]) # A list of one item, which is a dictionary.
    
    # This returns a list of images, even with 1 image id param, so we access one more time.
    img = img[0]

    img_filename = img['file_name']
    I = io.imread('%s/%s' % (img_dir, img_filename)) # This is an ndarray
    plt.axis('off')
    plt.imshow(I)
    # plt.show()
    print("Saving the image of a person.")
    plt.imsave(join(out_dir, 'person_img.png'), I)

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
    np.save(join(out_dir, "person_mask.npy"), mask)
    print("Saving the mask as an image too.")
    maskimg = Image.fromarray(np.uint8(mask * 255), 'L')
    maskimg.save(join(out_dir, 'mask.png'), 'PNG') 

if __name__ == "__main__":
    # If necessary we can have params in the future
    params  = ()
    locals_ = locals()
    fclass  = type(lambda: 1)
    funcs = [name for name in locals_ if type(locals_[name]) == fclass]

    print("Functions are: %s" % str(funcs))
    fname = input("Function to run: ")
    locals()[fname](*params)
