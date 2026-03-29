from copy import deepcopy

import numpy as np

from backbones.addmnist_joint import MNISTPairsDecoder, MNISTPairsEncoder
from backbones.addmnist_single import MNISTSingleEncoder
from backbones.disjointmnistcnn import DisjointMNISTAdditionCNN
from backbones.mnistcnn import MNISTAdditionCNN
from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST
from sumparity_split import in_distribution_mask, ood_mask


def _apply_mask(dataset, mask: np.ndarray):
    dataset.data = dataset.data[mask]
    dataset.concepts = dataset.concepts[mask]
    dataset.real_concepts = dataset.real_concepts[mask]
    dataset.targets = np.asarray(dataset.targets)[mask]
    return dataset


class SUMPARITYMNIST(BaseDataset):
    NAME = "sumparitymnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup, which_c=self.args.which_c, args=self.args
        )

        ood_test = self.get_ood_test(dataset_test)

        dataset_train, dataset_val, dataset_test = self.filtrate(
            dataset_train, dataset_val, dataset_test
        )

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.ood_test = ood_test

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(dataset_val, self.args.batch_size, val_test=True)
        self.test_loader = get_loader(dataset_test, self.args.batch_size, val_test=True)
        self.ood_loader = get_loader(ood_test, self.args.batch_size, val_test=True)

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.joint:
            if self.args.backbone == "neural":
                return MNISTAdditionCNN(), None
            return MNISTPairsEncoder(), MNISTPairsDecoder()

        if self.args.backbone == "neural":
            return DisjointMNISTAdditionCNN(n_images=self.get_split()[0]), None

        return MNISTSingleEncoder(), MNISTPairsDecoder()

    def get_split(self):
        if self.args.joint:
            return 1, (10, 10)
        return 2, (10,)

    def get_concept_labels(self):
        return [str(i) for i in range(10)]

    def get_labels(self):
        return ["even", "odd"]

    def filtrate(self, train_dataset, val_dataset, test_dataset):
        return (
            _apply_mask(train_dataset, in_distribution_mask(train_dataset.real_concepts)),
            _apply_mask(val_dataset, in_distribution_mask(val_dataset.real_concepts)),
            _apply_mask(test_dataset, in_distribution_mask(test_dataset.real_concepts)),
        )

    def get_ood_test(self, test_dataset):
        ood_test = deepcopy(test_dataset)
        return _apply_mask(ood_test, ood_mask(ood_test.real_concepts))

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test ID samples", len(self.dataset_test.data))
        print("Test OOD samples", len(self.ood_test.data))
