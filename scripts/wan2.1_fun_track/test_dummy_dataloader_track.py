#!/usr/bin/env python3
import argparse

import torch
from torch.utils.data import DataLoader

from videox_fun.data import DummyTrackLatentDataset


def collate_dummy(examples):
    latents = torch.stack([x["latents"] for x in examples], dim=0)
    texts = [x["text"] for x in examples]
    tracks = torch.stack([x["track_condition"]["tracks"] for x in examples], dim=0)
    visibility = torch.stack([x["track_condition"]["visibility"] for x in examples], dim=0)
    return {
        "latents": latents,
        "text": texts,
        "track_condition": {
            "tracks": tracks,
            "visibility": visibility,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Dummy dataloader smoke test for track training.")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--n_frames", type=int, default=81)
    parser.add_argument("--n_points", type=int, default=50)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--latent_frames", type=int, default=21)
    parser.add_argument("--latent_h", type=int, default=60)
    parser.add_argument("--latent_w", type=int, default=104)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = DummyTrackLatentDataset(
        length=args.length,
        latent_shape=(args.channels, args.latent_frames, args.latent_h, args.latent_w),
        n_frames=args.n_frames,
        n_points=args.n_points,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dummy,
    )

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        print(
            f"[step {step}] latents={tuple(batch['latents'].shape)} "
            f"tracks={tuple(batch['track_condition']['tracks'].shape)} "
            f"visibility={tuple(batch['track_condition']['visibility'].shape)} "
            f"texts={len(batch['text'])}"
        )


if __name__ == "__main__":
    main()
