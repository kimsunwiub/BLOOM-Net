# BLOOM-Net
Source code for "BLOOM-Net: Blockwise Optimization for Masking Networks Toward Scalable and Efficient Speech Enhancement" [1].

## Description

```e2e_train.py``` trains the masking network in the conventional end-to-end manner. A 1-layered model needs to be trained with this script initially before proceeding to the sequential training phase. 

```cont_train.py``` takes in previous model, inserts and trains an additional block while freezing all previous blocks and encoder module.

## Usage
For training E2E models, the e2e_train.py script can be run through

```python e2e_train.py --device 2 --is_ctn --nreps 2 --nblks 1 --tot_epoch 300 --duration 1 --no_skip_chan -b 64 --is_save```

To proceed training the next sequential block, 

```python cont_train.py --device 3 --nreps 6 --load_SEmodel prev_model_dir/model_last.pt --load_SErundata prev_model_dir/rundata_last.pt --is_save```

### Notes
- Number of epochs required for convergence of additional blocks may vary during sequential training. 

## References
1. S. Kim and M. Kim, “Bloom-net: Blockwise optimization for masking networks toward scalable andefficient speech enhancement (under review),” in Proc. of the IEEE International Conference on Acoustics,Speech, and Signal Processing (ICASSP), 2021.
