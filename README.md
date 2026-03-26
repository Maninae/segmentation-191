# Diamondback — Human Body Segmentation

Human body segmentation for CS 191 (Senior Research Project).

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `sample-backpack/` | Sample images from MS COCO |
| `scripts/` | Scripts for testing, data processing, regularization, etc. |
| `assets/` | Poster, images, and resources describing this project |
| `assets/output_imgs/` | Predicted segmentation masks for COCO samples in `sample-backpack/` |
| `model/` | Diamondback model architecture, losses, DenseNet encoder |
| `logs/` | Keras training logs over 19 epochs (zipped) |
| `util/` | Data generators, common paths |
| `weights/` | Location where Diamondback model weights are stored and loaded from |
| `tmp/` | Misc directory for debugging, used in test scripts |

---

## Usage

Download the model weights (not in GitHub because of size):

```bash
python3 download_diamondback_weights.py
# -> weights/diamondback_{...}.h5
# -> model/densenet_encoder/encoder_model.h5
```

Run a demo prediction:

```bash
python3 predict.py --demo --load_path weights/diamondback_{...}.h5
```

Train:

```bash
python3 training.py [--debug] [--load_path weights/diamondback_{...}.h5]
```

> [!NOTE]
> To run certain data scripts, make sure to install (or be in a virtualenv with) the COCO API.
> Some paths (namely `util/pathutil.py`, shell scripts in `tmp/`) are hardcoded for FloydHub's environment, where data and model weights were expected to be at the drive root `/`.

---

## Diamondback Model Overview

Inspiration mostly from Fu et al. in [SDN for Semantic Segmentation](https://arxiv.org/abs/1708.04943) (2017).
We trained Diamondback M2, which is two encoder-decoder units. More units add parameters but may improve results.

- Each unit employs dense convolutional connections.
- **Inter-unit connections**: Decoder feature maps at unit N-1 are concat'd with the encoder feature maps of the same resolution at unit N.
- **DenseNet encoder**: We transfer learn by using DenseNet-161's layers as the first unit's encoder.
  - 28x28 and 56x56 feature maps from the DN encoder are convolved and concat'd to every other unit's decoder layers of the respective resolution.
- **Using all units' learned decoder outputs**: We concatenate all units' decoder outputs (56x56), then upsample up to 224x224 for a final prediction tensor with 2 channels.

<img src="assets/diamondback_architecture.png" width="825"/>

---

## Training History

Losses and IOU over 19 epochs of training.

<img src="assets/diamondback_loss.png" width="400"/> <img src="assets/diamondback_IOU.png" width="400"/>

---

## Sample Predictions

| Input | Prediction | Ground Truth |
|:-----:|:----------:|:------------:|
| <img src="sample-backpack/people-val/000000100238.jpg" width="250"/> | <img src="assets/output_imgs/000000100238-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000100238-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000100510.jpg" width="250"/> | <img src="assets/output_imgs/000000100510-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000100510-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000100624.jpg" width="250"/> | <img src="assets/output_imgs/000000100624-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000100624-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000100723.jpg" width="250"/> | <img src="assets/output_imgs/000000100723-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000100723-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000200421.jpg" width="250"/> | <img src="assets/output_imgs/000000200421-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000200421-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000200839.jpg" width="250"/> | <img src="assets/output_imgs/000000200839-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000200839-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000200961.jpg" width="250"/> | <img src="assets/output_imgs/000000200961-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000200961-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000300276.jpg" width="250"/> | <img src="assets/output_imgs/000000300276-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000300276-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000300341.jpg" width="250"/> | <img src="assets/output_imgs/000000300341-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000300341-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000400044.jpg" width="250"/> | <img src="assets/output_imgs/000000400044-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000400044-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000400573.jpg" width="250"/> | <img src="assets/output_imgs/000000400573-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000400573-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000400803.jpg" width="250"/> | <img src="assets/output_imgs/000000400803-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000400803-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000400815.jpg" width="250"/> | <img src="assets/output_imgs/000000400815-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000400815-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000500270.jpg" width="250"/> | <img src="assets/output_imgs/000000500270-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000500270-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000500478.jpg" width="250"/> | <img src="assets/output_imgs/000000500478-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000500478-mask.png" width="250"/> |
| <img src="sample-backpack/people-val/000000500565.jpg" width="250"/> | <img src="assets/output_imgs/000000500565-pred.png" width="250"/> | <img src="sample-backpack/mask-val/000000500565-mask.png" width="250"/> |
