#!/bin/bash
floyd run --data ojwang/datasets/coco-people/1:/coco --env tensorflow-1.5 --tensorboard "python3 training.py"
