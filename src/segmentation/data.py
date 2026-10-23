"""Data loading for the Medical Segmentation Decathlon Task01_BrainTumour.

Each case has 4 MRI modalities (FLAIR, T1w, T1gd, T2w) and a label map with
classes: 0 = background, 1 = edema, 2 = non-enhancing tumour, 3 = enhancing tumour.
"""
from torch.utils.data import Subset

from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

TASK = "Task01_BrainTumour"
NUM_CLASSES = 4
KEYS = ["image", "label"]


def get_transforms(roi_size):
    """Return (train_transforms, val_transforms)."""
    base = [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        # z-score each modality separately, using only non-zero (brain) voxels
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    train = Compose(
        base
        + [
            # random patches, half centred on tumour, half on background
            RandCropByPosNegLabeld(
                keys=KEYS,
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=KEYS),
        ]
    )
    val = Compose(base + [EnsureTyped(keys=KEYS)])
    return train, val


def get_loaders(
    data_dir,
    roi_size=(96, 96, 96),
    batch_size=1,
    download=False,
    cache_rate=0.0,
    num_workers=2,
    max_train=None,
    max_val=None,
):
    """Build train/validation DataLoaders (80/20 split done by DecathlonDataset).

    max_train / max_val limit the number of cases, useful for quick test runs.
    Note: each training case yields 2 patches, so the effective batch is batch_size * 2.
    """
    train_tf, val_tf = get_transforms(roi_size)

    train_ds = DecathlonDataset(
        root_dir=data_dir,
        task=TASK,
        section="training",
        transform=train_tf,
        download=download,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = DecathlonDataset(
        root_dir=data_dir,
        task=TASK,
        section="validation",
        transform=val_tf,
        download=False,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    if max_train:
        train_ds = Subset(train_ds, range(min(max_train, len(train_ds))))
    if max_val:
        val_ds = Subset(val_ds, range(min(max_val, len(val_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader