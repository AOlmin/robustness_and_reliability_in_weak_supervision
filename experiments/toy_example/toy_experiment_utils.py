"""Utils for toy experiments"""
import torch
import torch.utils as torch_utils

from src.models.discriminative_model import DiscriminativeModel
from src.models.discriminative_model_mae import DiscriminativeModelMAE
from src.dataloaders.simple_classification_data import SimpleClassificationData

from src import neural_network_utils as nn_utils
from src import utils as custom_utils


def get_model(model_name, args):
    if model_name in custom_utils.available_discriminative_models():
        return get_discriminative_model(model_name, args)
    else:
        print("Model not found")


def get_discriminative_model(model_name, args):

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_classes = 2

    input_dim = 2
    model_layers = nn_utils.small_nn(input_size=input_dim, output_size=num_classes)

    if model_name == "standard_discriminative_model":
        model = DiscriminativeModel(model_layers, learning_rate=args.lr, device=device)
    elif model_name == "discriminative_model_mae":
        model = DiscriminativeModelMAE(model_layers, learning_rate=args.lr, device=device)
    else:
        print("Model not available")
        model = None

    return model


def train_supervised_model(model_name, args):

    train_data = SimpleClassificationData(store_file=args.data_dir + "/toy_data_train_5000_label_noise_"
                                                     + str(args.label_noise),
                                          reuse_data=True, num_samples=5000, label_noise=args.label_noise,
                                          random_state=1)

    train_loader = torch_utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    validation_data = SimpleClassificationData(store_file=args.data_dir + "/toy_data_valid_1000_label_noise_"
                                                          + str(args.label_noise),
                                               reuse_data=True, num_samples=1000, label_noise=args.label_noise,
                                               random_state=2)

    validation_loader = torch_utils.data.DataLoader(validation_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers)

    model = get_model(model_name, args)

    model.train_model(train_loader, validation_loader=validation_loader, num_epochs=args.num_epochs,
                      model_dir=args.model_dir, early_stopping=args.early_stopping)

    torch.save(model.state_dict(), args.model_dir + "/" + model.name + "_noise_" + str(args.label_noise))



