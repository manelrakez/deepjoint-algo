from deepjoint_torch import ROOT_DIR

IMAGE_HEIGHT = 576
IMAGE_WIDTH = 416
BEST_CHECKPOINT = ROOT_DIR / "models" / "torch_model" / "averaged_weights_valid_dense_loss.ckpt"
PRETRAINED_BREAST_MODEL = ROOT_DIR / "models" / "torch_model" / "pretrained_weights" / "breast_model.pth"
PRETRAINED_DENSE_MODEL = ROOT_DIR / "models" / "torch_model" / "pretrained_weights" / "dense_model.pth"
