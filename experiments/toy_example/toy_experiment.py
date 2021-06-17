import logging
from datetime import datetime
from pathlib import Path
import tikzplotlib

import numpy as np
import torch
import torch.utils as torch_utils
import matplotlib.pyplot as plt
from matplotlib import cm

from src import utils as custom_utils
from src.dataloaders.simple_classification_data import SimpleClassificationData
import experiments.toy_example.toy_experiment_utils as tailored_utils

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOGGER = logging.getLogger(__name__)


def label_noise_experiment(args):
    """Train and evaluate model on circle data"""
    tailored_utils.train_supervised_model("standard_discriminative_model", args)
    test_model("standard_discriminative_model", args)


def test_model(model_type, args):
    model = tailored_utils.get_model(model_type, args)

    state_dict = torch.load(args.model_dir + "/" + model.name + "_noise_" + str(args.label_noise),
                            map_location=model.device)
    model.load_state_dict(state_dict)

    model.eval()

    # Evaluate model on train data
    train_data = SimpleClassificationData(store_file=args.data_dir + "/toy_data_train_5000_label_noise_"
                                                     + str(args.label_noise),
                                          reuse_data=True, num_samples=5000, label_noise=args.label_noise,
                                          random_state=1)
    train_loader = torch_utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    train_acc = model.get_accuracy(train_loader).to(torch.device("cpu")).data.numpy()
    LOGGER.info("Accuracy on train data is {}".format(train_acc))

    # Evaluate model on test data
    test_data = SimpleClassificationData(store_file=args.data_dir + "/toy_data_test_1000", reuse_data=True,
                                         num_samples=1000, label_noise=0, random_state=3)

    test_loader = torch_utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    test_acc = model.get_accuracy(test_loader).to(torch.device("cpu")).data.numpy()
    LOGGER.info("Accuracy on test data is {}".format(test_acc))

    # Plot train data
    fig, ax = plt.subplots()
    plot_data(train_data.get_full_data(), ax)

    # Visualise model predictions (model uncertainty)
    fig, ax = plt.subplots()
    distribution_plot(model, fig, ax)

    tikzplotlib.save("experiments/toy_example/figures/uncertainty_plot_noise_" + str(args.label_noise) + ".tex")
    plt.show()


def plot_data(data, ax):
    """Plot two-dimensional data"""
    inputs = data[:, :-2]
    labels = data[:, -2:]
    label_1_inds = labels[:, 0] == 1
    label_2_inds = labels[:, 1] == 1
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    ax.scatter(inputs[label_1_inds, 0], inputs[label_1_inds, 1], c="#FDE725FF", s=3)
    ax.scatter(inputs[label_2_inds, 0], inputs[label_2_inds, 1], c="#440154FF",  s=3)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.legend(["Class 1", "Class 2"])


def distribution_plot(model, fig, ax):
    # Look at distributions over input space

    X1 = np.arange(-1.5, 1.5, 0.01)
    X2 = np.arange(-1.5, 1.5, 0.01)
    X1, X2 = np.meshgrid(X1, X2)

    tensor_input = torch.tensor(np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=-1), dtype=torch.float)
    Y = model.predict(tensor_input).data.numpy().reshape(X1.shape[0], X1.shape[1], -1)

    p_min = 0.0
    p_max = 1.0
    surf = ax.pcolormesh(X1, X2, Y[:, :, 0], cmap="coolwarm", shading='auto', linewidth=0, antialiased=False,
                         vmin=p_min, vmax=p_max)

    fig.colorbar(surf, ax=ax)

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("p(y=1 | x)")


if __name__ == '__main__':
    args = custom_utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    custom_utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                              log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))

    label_noise_experiment(args)

