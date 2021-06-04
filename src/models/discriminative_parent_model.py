from abc import ABC, abstractmethod
import logging

import torch
import torch.nn as nn

import src.utils as custom_utils


class DiscriminativeParentModel(nn.Module, ABC):
    """Parent class for discriminative models"""

    def __init__(self, loss, device=torch.device("cpu")):
        super().__init__()

        self._log = logging.getLogger(self.__class__.__name__)
        self.loss = loss
        self.device = device
        self._log.info("Moving model to {}".format(device))

    def train_model(self, train_loader, validation_loader=None, num_epochs=100, model_dir="/models",
                    early_stopping=False, save_interval=1):
        """Train model"""

        if early_stopping:
            # Save best model during training
            torch.save(self.state_dict(), model_dir + "/" + self.name + "_saved_best")
            best_epoch = 0

            # Set initial loss to high value
            opt_loss = 10000

        # Save initial model
        torch.save(self.state_dict(), model_dir + "/" + self.name + "_" + str(0))

        running_loss = torch.zeros((num_epochs,))
        val_loss = torch.zeros((num_epochs,))

        for epoch in range(num_epochs):
            running_loss[epoch] = self._train_epoch(train_loader)

            self._log.info("Learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
            self._log.info("Running loss, epoch {}: {}".format(epoch + 1, running_loss[epoch]))

            if validation_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss[epoch] = self._validate(validation_loader)

                new_loss = val_loss[:(epoch+1)]

                self._log.info("Validation loss: {}".format(val_loss[epoch]))
                self.train_mode()
            else:
                new_loss = running_loss.detach()[:(epoch+1)]

            if early_stopping:
                # Check if early stopping condition is fulfilled (use validation dataloader in first place)
                if self._early_stopping_condition(new_loss):
                    self._log.info("Early stopping conditioned fulfilled")
                    self._log.info("Loading model from epoch {}".format(best_epoch + 1))
                    saved_state_dict = torch.load(model_dir + "/" + self.name + "_saved_best")
                    self.load_state_dict(saved_state_dict)
                    break
                else:
                    if new_loss[epoch] < opt_loss:
                        opt_loss = new_loss[epoch]
                        best_epoch = epoch
                        torch.save(self.state_dict(), model_dir + "/" + self.name + "_saved_best")

            if torch.isnan(running_loss[epoch]):
                break

            # Save model every epoch as default
            if (epoch + 1) % save_interval == 0:
                torch.save(self.state_dict(), model_dir + "/" + self.name + "_" + str(epoch + 1))
                self._log.info("Saving model epoch {}".format(epoch + 1))

        return running_loss

    def _train_epoch(self, train_loader):
        """Train epoch"""
        running_loss = 0
        for batch in train_loader:

            self.optimizer.zero_grad()
            inputs, targets = batch

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            targets = self._modify_targets(targets, inputs)
            outputs = self._forward(inputs)

            loss = self._calculate_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    def _validate(self, validation_loader):
        """Calculate (running) loss over validation loader"""
        val_loss = 0
        for batch in validation_loader:
            self.optimizer.zero_grad()
            inputs, targets = batch

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            targets = self._modify_targets(targets, inputs)
            outputs = self._forward(inputs)

            val_loss += self._calculate_loss(outputs, targets)

        return val_loss

    def _modify_targets(self, targets, inputs):
        """Modify targets prior to loss calculation"""
        return targets

    def _calculate_loss(self, outputs, targets):
        """Calculate loss"""
        outputs = self._output_transform(outputs)
        return self.loss(outputs, targets)

    def _predict_transform(self, output):
        """Transformation of output for prediction"""
        return output

    def _output_transform(self, output):
        """Transformation of output for calculation of loss"""
        return output

    def train_mode(self, mode=True):
        self.training = mode

        for module in self.children():
            module.train(mode)

    def eval(self):
        self.train_mode(mode=False)

    def _early_stopping_condition(self, new_loss, alpha=0.1):
        return custom_utils.generalisation_loss_condition(new_loss, alpha=alpha)

    def predict(self, inputs):
        """Make (soft) prediction for given input"""
        inputs = inputs.to(self.device)
        return self._predict_transform(self._forward(inputs))

    def predict_labels(self, inputs):
        """Hard prediction for given input"""
        inputs = inputs.to(self.device)
        return torch.argmax(self._predict_transform(self._forward(inputs)), dim=-1)

    def predict_raw(self, inputs):
        """Predict without final transformation"""
        return self._forward(inputs)

    def get_accuracy(self, data_loader):
        """Predict and calculate accuracy over dataloader"""
        provided_labels = []
        predictions = []
        for batch in data_loader:
            inputs, labels = batch
            labels = labels.to(self.device)
            provided_labels.append(torch.argmax(labels, dim=1))

            predictions.append(self.predict_labels(inputs))

        return (torch.cat(predictions) == torch.cat(provided_labels)).float().mean()

    @abstractmethod
    def _forward(self, inputs):
        pass





