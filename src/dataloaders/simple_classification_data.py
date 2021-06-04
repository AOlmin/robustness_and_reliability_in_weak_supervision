# Dataloader for circle data

from pathlib import Path
import csv
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn import datasets


class SimpleClassificationData(torch.utils.data.Dataset):
    def __init__(self, store_file, reuse_data=False, num_samples=1000, label_noise=0.0, random_state=1):

        super().__init__()

        self.num_classes = 2

        self.num_samples = num_samples
        self.label_noise = label_noise
        self.random_state = random_state

        # Load data if it exists
        self.file = Path(store_file)
        if self.file.exists() and reuse_data:
            data = self.get_full_data()
            self.x, self.y = data[:, :-2], data[:, -2:]
            self.validate_dataset()
        else:
            # Sample new data
            print("Sampling new data")
            self.x, self.y = self.sample_new_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        inputs, labels = self.x[index], self.y[index]
        return (np.array(inputs,
                         dtype=np.float32), np.array(labels, dtype=np.float32))

    def sample_new_data(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)

        x, y = datasets.make_circles(n_samples=self.num_samples, factor=.5, noise=0.1, random_state=self.random_state)
        y = y.reshape(-1, 1)

        # label flips
        if self.label_noise != 0:
            flip_sample = np.random.uniform(size=(self.num_samples, 1))
            y[flip_sample <= self.label_noise] = 1 - y[flip_sample <= self.label_noise]

        # Create one-hot targets
        y = (np.eye(self.num_classes)[y.astype(np.long)[:, 0]]).astype(np.float32)

        # Combine and save data
        combined_data = np.column_stack((x, y))
        np.random.shuffle(combined_data)
        np.savetxt(self.file, combined_data, delimiter=",")

        return x, y

    def validate_dataset(self):
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            assert self.num_samples == sum(1 for _ in csv_reader)

    def get_full_data(self, type_tensor=False):
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]

        data = np.array(tmp_raw_data, dtype=float)

        if type_tensor:
            data = torch.tensor(data)

        return data


def plot_2d_data(data, ax):
    inputs = data[:, :-2]
    labels = data[:, -2:]
    label_1_inds = labels[:, 0] == 1
    label_2_inds = labels[:, 1] == 1
    ax.scatter(inputs[label_1_inds, 0], inputs[label_1_inds, 1], s=3)
    ax.scatter(inputs[label_2_inds, 0], inputs[label_2_inds, 1], s=3)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    ax.legend(["Class 0", "Class 1"])


def main():
    """Entry point for debug visualisation"""
    _, ax = plt.subplots()

    dataset = SimpleClassificationData(store_file=Path("data/toy_2_data_1000"), num_samples=1000, label_noise=0.3)

    plot_2d_data(dataset.get_full_data(), ax)
    plt.show()


if __name__ == "__main__":
    main()
