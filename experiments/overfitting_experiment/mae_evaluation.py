import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils as torch_utils
import matplotlib.pyplot as plt

from src import utils as custom_utils
from src.dataloaders.mnist_data import MnistData
import experiments.overfitting_experiment.mnist_experiment_utils as tailored_utils
import tikzplotlib

LOGGER = logging.getLogger(__name__)


def main(args):
    step = args.save_interval

    if args.init_model is not None:
        id = "_pretrained"
    else:
        id = ""

    model = tailored_utils.get_model("discriminative_model_mae", args)

    LOGGER.info("Label noise {}".format(args.label_noise))
    LOGGER.info("Loading models from {}".format(args.model_dir))

    ind = np.load(args.data_dir + "/training_indices.npy")
    split = 50000

    train_data = MnistData(ind=ind[:split], root=args.data_dir, label_noise=args.label_noise, reuse_targets=True)

    train_loader = torch_utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers)

    test_data = MnistData(train=False, label="test", root=args.data_dir)
    test_loader = torch_utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    train_acc_list = []
    mae_loss_list = []
    test_acc_list = []

    # Loop over all epochs
    for i in range(int(args.num_epochs / step) + 1):
        LOGGER.info("Model {}".format(i*step))
        state_dict = torch.load(args.model_dir + "/" + model.name + "_" + str(i*step), map_location=model.device)
        model.load_state_dict(state_dict)
        model.eval()

        # Evaluate model on train data
        train_predictions, train_targets = custom_utils.iter_function(train_loader, model.predict,
                                                                      device=model.device, return_targets=True)

        train_acc = (torch.argmax(train_predictions, dim=-1) == torch.argmax(train_targets, dim=-1)).float().mean()
        mae_loss = model.loss(train_predictions, train_targets) * 10  # Scale with K=10 to get sum over classes

        LOGGER.info("Accuracy on train data is {}".format(train_acc))
        LOGGER.info("Loss on train data is {}".format(mae_loss))

        # Evaluate model on test data
        test_acc = model.get_accuracy(test_loader).to(torch.device("cpu")).data.numpy()
        LOGGER.info("Accuracy on test data is {}".format(test_acc))

        train_acc_list.append(train_acc)
        mae_loss_list.append(mae_loss)
        test_acc_list.append(test_acc)

    train_acc = np.array(train_acc_list)
    train_loss = np.array(mae_loss_list)
    test_acc = np.array(test_acc_list)
    epochs = np.arange(0, args.num_epochs + 1, step)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, train_acc)
    ax[0].plot(epochs, test_acc)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    #ax[0].set_ylim([0.5, 1.1])
    ax[0].legend(["Train", "Test"])

    ax[1].plot(epochs, train_loss)
    ax[1].set_xlabel("Epoch")
    #ax[1].set_ylim([0.5, 1.1])
    ax[1].set_ylabel("MAE")

    tikzplotlib.save("experiments/overfitting_experiment/figures/MAE_test_noise" + id + "_" + str(args.label_noise)
                     + ".tex")

    plt.show()


if __name__ == '__main__':
    args = custom_utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    custom_utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                              log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))

    main(args)

