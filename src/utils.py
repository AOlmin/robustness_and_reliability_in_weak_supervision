import sys
import logging
import argparse
from pathlib import Path
import torch


def generalisation_loss_condition(val, alpha=0.1):
    """Generalisation loss condition for early stopping
    :param val: vector of metric values up to current epoch
    :param alpha: (float) maximum improvement to continue training
    :returns: True if training should stop, False otherwise
    """

    gl = val[-1] / val.min() - 1
    return gl > alpha


def iter_function(data_loader, func, device=torch.device("cpu"), target_as_input=False, return_targets=False):
    """Helper to perform function over dataloader"""
    res = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            input, target = batch
            input, target = input.to(device), target.to(device)

            if target_as_input:
                output = func(input, target)
            else:
                output = func(input)

            res.append(output)
            targets.append(target)

    if return_targets:
        return torch.cat(res), torch.cat(targets)
    else:
        return torch.cat(res)


def parse_args():
    """Arg parser"""
    parser = argparse.ArgumentParser(description="MODEL_ARGUMENTS")
    parser.add_argument("--batch_size", type=int,
                        default=100,
                        help="Batch size for data loaders")
    parser.add_argument("--data_dir",
                        default="./data",
                        help="Data directory")
    parser.add_argument("--early_stopping",
                        action="store_true",
                        help="Use early stopping")
    parser.add_argument("--gpu",
                        action="store_true",
                        help="Use gpu, if available")
    parser.add_argument("--init_model",
                        default=None,
                        help="Directory to pre-trained model (random initialisation if None)")
    parser.add_argument("--label_noise", type=float,
                        default=0.0,
                        help="Proportion of label noise in labelled data")
    parser.add_argument("--log_dir",
                        type=Path,
                        default="./logs",
                        help="Log directory")
    parser.add_argument("--log_level",
                        type=_log_level_arg,
                        default=logging.INFO,
                        help="Log level")
    parser.add_argument("--lr", type=float,
                        default=0.1,
                        help="Learning rate")
    parser.add_argument("--model_dir",
                        default="./models",
                        help="Model directory")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=500,
                        help="Number of epochs for training")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers in data loader")
    parser.add_argument("--save_interval",
                        type=int,
                        default=1,
                        help="At what to save intermediate models during training")
    return parser.parse_args()


def _log_level_arg(arg_string):
    """Log arguments"""
    arg_string = arg_string.upper()
    if arg_string == "DEBUG":
        log_level = logging.DEBUG
    elif arg_string == "INFO":
        log_level = logging.INFO
    elif arg_string == "WARNING":
        log_level = logging.WARNING
    elif arg_string == "ERROR":
        log_level = logging.WARNING
    elif arg_string == "CRITICAL":
        log_level = logging.WARNING
    else:
        raise argparse.ArgumentTypeError(
            "Invalid log level: {}".format(arg_string))
    return log_level


def setup_logger(log_path=None,
                 logger=None,
                 log_level=logging.INFO,
                 fmt="%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s"):
    """Setup for a logger instance.

    Args:
        log_path (str, optional): full path
        logger (logging.Logger, optional): root logger if None
        log_level (logging.LOGLEVEL, optional):
        fmt (str, optional): message format

    """
    logger = logger if logger else logging.getLogger()
    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.setLevel(log_level)
    logger.handlers = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    log_path = Path(log_path)
    if log_path:
        directory = log_path.parent
        directory.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Log at {}".format(log_path))


def available_discriminative_models():
    """Helper for checking that model exists"""
    return ["standard_discriminative_model", "discriminative_model_mae"]
