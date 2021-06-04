import logging
from datetime import datetime
from pathlib import Path

from src import utils as custom_utils
import experiments.overfitting_experiment.mnist_experiment_utils as tailored_utils
from experiments.overfitting_experiment import cce_experiment

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOGGER = logging.getLogger(__name__)


def label_noise_experiment(args):
    """Train discriminative model using MAE loss"""

    if args.init_model is not None:
        LOGGER.info("Initialising model from: {}".format(args.init_model))
    tailored_utils.train_supervised_model("discriminative_model_mae", args)
    cce_experiment.test_model("discriminative_model_mae", args)


if __name__ == '__main__':
    args = custom_utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    custom_utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                              log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))

    label_noise_experiment(args)

