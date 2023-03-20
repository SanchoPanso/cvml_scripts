import os
import sys
import glob
import cv2
import numpy as np
import torch
import logging
from typing import Callable, List
import argparse
from pathlib import Path
import zipfile
from filesplit.split import Split

import cvml
from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.dataset.image_transforming import normalize_min_max, SPEstimatorNumpy, expo

print(torch.cuda.is_available())

img = cv2.imread('/home/student2/datasets/raw/csv1_comet_1/images/72.png')
exp_img = expo(img, 15)

cv2.imshow('img', cv2.resize(img, (400, 400)))
cv2.imshow('exp_img', cv2.resize(exp_img, (400, 400)))
cv2.waitKey()