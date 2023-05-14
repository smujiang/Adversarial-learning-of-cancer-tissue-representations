#!/usr/bin/bash
python project_real_tissue_latent_space.py --batch_size 8 --z_dim 200 --checkpoint /infodev1/non-phi-data/junjiang/Path_GAN/pretrained_model/vgh_nki/checkpoints/PathologyGAN_Encoder.ckt --real_hdf5 /infodev1/non-phi-data/junjiang/Path_GAN/example_data/vgh_nki/he/patches_h224_w224/hdf5_vgh_nki_he_test.h5 --batch_size 4 --main_path /infodev1/non-phi-data/junjiang/Path_GAN/output_test --features



