#!/bin/bash
floyd run \
    --data ojwang/datasets/coco-people/1:/coco \
    --data ojwang/datasets/diamondback_ep07/3:/weights \
    --env tensorflow-1.5 \
    --gpu+ \
    --tensorboard \
    "python3 training.py --load_path /weights/diamondback_ep07-tloss=42396.4888-vloss=47624.5262-tIOU=0.7504-vIOU=0.7310_WITHREG.h5"
