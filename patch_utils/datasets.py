import torch
from utils.datasets import LoadImagesAndLabels


def load_dataset(imgs_path, img_size, batch_size=1, n_cpu=1):
    dataset = LoadImagesAndLabels(
        imgs_path,
        img_size,
        batch_size,
        augment=False,  # augment images
        hyp=None,  # TODO: augmentation hyperparameters
        rect=False,  # rectangular training
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        image_weights=False,
        prefix="",
    ) 

    batch_size = min(batch_size, len(dataset))
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=n_cpu,
        sampler=None,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader
