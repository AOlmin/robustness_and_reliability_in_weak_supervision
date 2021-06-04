"""Data loader for MNIST data"""
import logging
import os

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class MnistData:
    """MNIST data wrapper
    """

    def __init__(self, train=True, label="train", augmentation=False, root="./data", label_noise=0.0, ind=None,
                 reuse_targets=False, flattened_x=True):

        self._log = logging.getLogger(self.__class__.__name__)

        self.num_classes = 10
        self.flattened_x = flattened_x

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()])

        else:
            self.transform = None

        self.set = torchvision.datasets.MNIST(root=root,
                                              train=train,
                                              download=True)

        self.set.data = np.array(self.set.data)
        self.set.targets = np.array(self.set.targets)

        if ind is not None:
            self.set.data = self.set.data[ind, :, :]
            self.set.targets = self.set.targets[ind]
            unique, counts = np.unique(self.set.targets, return_counts=True)
            print(dict(zip(unique, counts)))

        self.input_size = self.set.data.shape[0]

        # Even if we load targets or add noise to them, we want to store the true targets
        self.true_labels = self.set.targets.copy()

        if self.flattened_x:
            self.set.data = self.set.data.reshape(self.input_size, -1)

        if ind is not None:
            file_dir = root + "/MNIST/targets/" + label + "_" + str(ind.shape[0]) + "_targets_noise_level_" + str(label_noise) + ".npy"
        else:
            file_dir = root + "/MNIST/targets/" + label + "_targets_noise_level_" + str(label_noise) + ".npy"

        # Check if noisy targets already exist
        if reuse_targets and os.path.exists(file_dir):
            # Note: this is dependent upon that the data is always loaded in the same order
            self.set.targets = np.load(file_dir)
            self._log.info("Targets loaded from " + file_dir)

            if self.set.targets.shape[0] != self.set.data.shape[0] and ind is not None:
                self.set.targets = self.set.targets[ind]

            print("Proportion of correct labels: {}".format(np.mean(self.true_labels == self.set.targets)))
        else:
            if reuse_targets:
                self._log.info("Target file not found, targets not loaded")

            # Add label noise to data
            if label_noise != 0.0:

                epsilon = np.random.uniform(size=self.set.data.shape[0])
                samples_to_flip = (epsilon <= label_noise)
                for k in range(self.num_classes):
                    class_samples = self.true_labels == k
                    rel_samples = class_samples * samples_to_flip

                    other_classes = np.delete(np.arange(self.num_classes), k)
                    self.set.targets[rel_samples] = np.random.choice(other_classes, np.sum(rel_samples))

            np.save(file_dir, self.set.targets)
            print("Proportion of correct labels: {}".format(np.mean(self.true_labels == self.set.targets)))
            self._log.info("Saving targets to " + file_dir)

        # Convert targets to one hot
        self.set.targets = (np.eye(self.num_classes)[self.set.targets]).astype(np.float32)

    def __len__(self):
        return self.input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.set.data[index], self.set.targets[index]

        if self.flattened_x:
            img = torch.tensor(img / 255, dtype=torch.float32)

        else:
            img = Image.fromarray(img)  # This should scale images to [0, 1]

            if self.transform is not None:
                img = transforms.ToTensor()(self.transform(img))
            else:
                img = transforms.ToTensor()(img)

        target = torch.tensor(target)

        return img, target


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = MnistData()

    loader = torch.utils.data.DataLoader(data,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=0)
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # print labels
    print("Labels: {}".format(labels.data.numpy()))

    # show images
    plt.imshow(np.transpose(torchvision.utils.make_grid(images).numpy(), (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
