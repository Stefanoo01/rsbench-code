# CBM model for biased MNIST Sum-Parity
import torch
import torch.nn as nn

from models.utils.cbm_module import CBMModule
from utils.args import *
from utils.conf import get_device
from utils.dpl_loss import ADDMNIST_DPL
from utils.losses import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Learning via Concept Extractor.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistSumParityCBM(CBMModule):
    NAME = "mnistsumparitycbm"

    def __init__(
        self,
        encoder,
        n_images=2,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=20,
        nr_classes=2,
    ):
        super().__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        self.n_images = n_images
        self.c_split = c_split

        self.n_facts = 10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
        self.nr_classes = 2

        self.opt = None
        self.device = get_device()

        self.classifier = nn.Sequential(
            nn.Linear(self.n_facts * self.n_images, 32),
            nn.ReLU(),
            nn.Linear(32, self.nr_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])
            cs.append(lc)
        clen = len(cs[0].shape)

        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        pCs = self.normalize_concepts(cs)
        py = self.cmb_inference(cs)

        return {"CS": cs, "YS": py, "pCS": pCs}

    def cmb_inference(self, cs, query=None):
        flattened_cs = cs.view(cs.shape[0], cs.shape[1] * cs.shape[2])
        return self.classifier(flattened_cs)

    def normalize_concepts(self, z, split=2):
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        prob_digit1 = nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = nn.Softmax(dim=1)(prob_digit2)

        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / z1

        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / z2

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)

    @staticmethod
    def get_loss(args):
        if args.dataset in ["sumparitymnist"]:
            return ADDMNIST_DPL(ADDMNIST_Cumulative, nr_classes=2)
        raise NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )
