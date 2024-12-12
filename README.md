# Affine-Combining Autoencoder (ACAE)

This repository contains code for the paper "Learning 3D Human Pose Estimation from Dozens of Datasets using a Geometry-Aware Autoencoder to Bridge Between Skeleton Formats" by István Sárándi, Alexander Hermans, and Bastian Leibe.
The paper relies also on code from several other repos, including [metrabs](https://github.com/isarandi/metrabs) and [posepile](https://github.com/isarandi/posepile). This repo is specifically for the ACAE part. 

See the `train_acae` function in `acae.py` for how to train an ACAE.

## Publication reference
If you find this code useful, consider citing the paper:

```bibtex
@inproceedings{Sarandi2023acae,
    author = {S\'ar\'andi, Istv\'an and Hermans, Alexander and Leibe, Bastian},
    title = {Learning {3D} Human Pose Estimation from Dozens of Datasets using a Geometry-Aware Autoencoder to Bridge Between Skeleton Formats},
    booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year = {2023}
} 
```

## License
GNU Affero General Public License, Version 3 (AGPL-3.0)

## Legal disclaimer

This software is a research prototype only and shall only be used for test-purposes. This software must not be used in or for products and/or services and in particular not in or for safety-relevant areas. It was solely developed for and
published as part of the publication ‘Learning 3D Human Pose Estimation From
Dozens of Datasets Using a Geometry-Aware Autoencoder To Bridge Between
Skeleton Formats’ and will neither be maintained nor monitored in any way.
