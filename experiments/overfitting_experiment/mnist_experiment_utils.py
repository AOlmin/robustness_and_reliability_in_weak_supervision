"""Utils for MNIST experiments"""
import torch
import torch.utils as torch_utils
import numpy as np

from src.models.discriminative_model import DiscriminativeModel
from src.models.discriminative_model_mae import DiscriminativeModelMAE
from src.dataloaders.mnist_data import MnistData

from src import neural_network_utils as nn_utils
from src import utils as custom_utils


def get_model(model_name, args):
    """Check if model exists"""
    if model_name in custom_utils.available_discriminative_models():
        return get_discriminative_model(model_name, args)
    else:
        print("Model not found")


def get_discriminative_model(model_name, args):
    """Initialise model"""

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_classes = 10
    input_dim = 28 ** 2

    model_layers = nn_utils.simple_nn(input_size=input_dim, output_size=num_classes)

    if model_name == "standard_discriminative_model":
        model = DiscriminativeModel(model_layers, learning_rate=args.lr, device=device)
    elif model_name == "discriminative_model_mae":
        model = DiscriminativeModelMAE(model_layers, learning_rate=args.lr, device=device)

    # Keep track of pre-trained models
    if args.init_model is not None:
        model.name += "_pretrained"

    return model


def train_supervised_model(model_name, args):

    ind = np.load(args.data_dir + "/training_indices.npy")
    split = 50000
    train_data = MnistData(ind=ind[:split], root=args.data_dir, label_noise=args.label_noise, reuse_targets=True)

    train_loader = torch_utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    validation_data = MnistData(ind=ind[split:], label="valid", root=args.data_dir, label_noise=args.label_noise,
                                reuse_targets=True)
    validation_loader = torch_utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers)

    model = get_model(model_name, args)

    # Option to initialise from specific model
    if args.init_model is not None:
        init_state_dict = torch.load(args.init_model)

        if init_state_dict is not None:
            print("Loading initial model")
            model.load_state_dict(init_state_dict)
        else:
            print("Could not load initial model")

    model.train_model(train_loader, validation_loader=validation_loader, num_epochs=args.num_epochs,
                      model_dir=args.model_dir, early_stopping=args.early_stopping)

    torch.save(model.state_dict(), args.model_dir + "/" + model.name + "_noise_" + str(args.label_noise))




