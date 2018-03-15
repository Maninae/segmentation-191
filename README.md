# segmentation-191
Human body segmentation project for CS 191 (Diamondback), Senior research project.

To run data scripts, make sure to install (or be in a virtualenv with) the COCO API.

See ```assets``` for poster describing this project in more detail.

-- ojwang@cs.stanford.edu

## Training History
Losses and IOU over 19 epochs of training.

<img src="assets/diamondback_loss.png" width="375"/> <img src="assets/diamondback_IOU.png" width="375"/>

## Sample Images
Columns left to right: (1) Image, (2) Diamondback M2 model's prediction, (3) Ground truth segmentation masks.

<img src="sample-backpack/people-val/000000100238.jpg" width="250"/> <img src="assets/output_imgs/000000100238-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000100238-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000100510.jpg" width="250"/> <img src="assets/output_imgs/000000100510-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000100510-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000100624.jpg" width="250"/> <img src="assets/output_imgs/000000100624-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000100624-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000100723.jpg" width="250"/> <img src="assets/output_imgs/000000100723-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000100723-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000200421.jpg" width="250"/> <img src="assets/output_imgs/000000200421-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000200421-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000200839.jpg" width="250"/> <img src="assets/output_imgs/000000200839-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000200839-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000200961.jpg" width="250"/> <img src="assets/output_imgs/000000200961-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000200961-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000300276.jpg" width="250"/> <img src="assets/output_imgs/000000300276-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000300276-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000300341.jpg" width="250"/> <img src="assets/output_imgs/000000300341-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000300341-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000400044.jpg" width="250"/> <img src="assets/output_imgs/000000400044-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000400044-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000400573.jpg" width="250"/> <img src="assets/output_imgs/000000400573-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000400573-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000400803.jpg" width="250"/> <img src="assets/output_imgs/000000400803-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000400803-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000400815.jpg" width="250"/> <img src="assets/output_imgs/000000400815-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000400815-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000500270.jpg" width="250"/> <img src="assets/output_imgs/000000500270-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000500270-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000500478.jpg" width="250"/> <img src="assets/output_imgs/000000500478-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000500478-mask.png" width="250"/>

<img src="sample-backpack/people-val/000000500565.jpg" width="250"/> <img src="assets/output_imgs/000000500565-pred.png" width="250"/> <img src="sample-backpack/mask-val/000000500565-mask.png" width="250"/>