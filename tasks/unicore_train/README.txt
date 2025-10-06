# This directory contains template code for training UniCore models.
1. First, please install uni-core via pip from https://github.com/dptech-corp/Uni-Core. (Note: uni-core has a memory leak issue.)
2. Please prepare your filtering dataset containing system and MSA features, and avoid using ligand datasets by modifying feature_loader_plinder.
3. This is only reference training code. We will soon release PhysDock 2.0 with lightning-based training code and validation settings.