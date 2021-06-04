import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils as torch_utils

from src import utils as custom_utils
from src.dataloaders.mnist_data import MnistData
import experiments.overfitting_experiment.mnist_experiment_utils as tailored_utils

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOGGER = logging.getLogger(__name__)


def main(args):
    """Train discriminative model using CCE loss"""
    tailored_utils.train_supervised_model("standard_discriminative_model", args)
    test_model("standard_discriminative_model", args)


def test_model(model_type, args):
    model = tailored_utils.get_model(model_type, args)

    state_dict = torch.load(args.model_dir + "/" + model.name + "_noise_" + str(args.label_noise),
                            map_location=model.device)
    model.load_state_dict(state_dict)

    model.eval()

    ind = np.load(args.data_dir + "/training_indices.npy")
    split = 50000

    # Evaluate performance on train data
    train_data = MnistData(ind=ind[:split], root=args.data_dir, label_noise=args.label_noise, reuse_targets=True)

    train_loader = torch_utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    train_acc = model.get_accuracy(train_loader).to(torch.device("cpu")).data.numpy()
    LOGGER.info("Accuracy on train data is {}".format(train_acc))

    # Evaluate performance on test data
    test_data = MnistData(train=False, label="test", root=args.data_dir)
    test_loader = torch_utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    test_acc = model.get_accuracy(test_loader).to(torch.device("cpu")).data.numpy()
    LOGGER.info("Accuracy on test data is {}".format(test_acc))


if __name__ == '__main__':
    args = custom_utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    custom_utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                              log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))

    main(args)

