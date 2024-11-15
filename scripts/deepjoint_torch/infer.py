import sys

from argparse import ArgumentParser
from pathlib import Path

from deepjoint_torch.log import set_logger
from deepjoint_torch.constants import BEST_CHECKPOINT
from deepjoint_torch.infer import infer




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("-h5", "--h5_dir", dest="h5_dir", type=Path, required=True)

    parser.add_argument("-anno", "--annotations", dest="annotations", type=Path, default=None)
    parser.add_argument("-ckpt", "--checkpoint", dest="checkpoint", type=Path, default=BEST_CHECKPOINT)
    parser.add_argument("--eval_model", action="store_true")

    params = parser.parse_args(sys.argv[1:])

    if not params.output_dir.exists():
        params.output_dir.mkdir(parents=True)

    set_logger(output_dir=params.output_dir)

    infer(
        output_dir=params.output_dir,
        h5_dir=params.h5_dir,
        checkpoint=params.checkpoint,
        annotations=params.annotations,
        eval_model=params.eval_model,
    )
