# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

from deepjoint_torch import ROOT_DIR

IMAGE_HEIGHT = 576
IMAGE_WIDTH = 416
BEST_CHECKPOINT = ROOT_DIR / "models" / "torch_model" / "averaged_weights_valid_dense_loss.ckpt"
PRETRAINED_BREAST_MODEL = ROOT_DIR / "models" / "torch_model" / "pretrained_weights" / "breast_model.pth"
PRETRAINED_DENSE_MODEL = ROOT_DIR / "models" / "torch_model" / "pretrained_weights" / "dense_model.pth"
