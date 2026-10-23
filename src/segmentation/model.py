"""Segmentation model for NeuroLens.

SegResNet is a compact 3D encoder-decoder that trains well on small GPUs.
Dropout is built in on purpose: keeping it active at inference time later
gives us Monte Carlo dropout uncertainty (milestone 2) with no model change.
"""
from monai.networks.nets import SegResNet

from .data import NUM_CLASSES


def build_model(in_channels=4, out_channels=NUM_CLASSES, dropout=0.2):
    return SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
    )