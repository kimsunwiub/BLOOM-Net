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

To fine-tune BLOOM-Net,

```python finetuning.py --device 1 --ori_l1_dir ori_l1_dir/model_last.pt --feat_seq_l2_dir feat_seq_l2_dir/model_last.pt --feat_seq_l3_dir feat_seq_l3_dir/model_last.pt --feat_seq_l4_dir feat_seq_l4_dir/model_last.pt --feat_seq_l5_dir feat_seq_l5_dir/model_last.pt --feat_seq_l6_dir feat_seq_l6_dir/model_last.pt --is_save``` 

### Notes
- Number of epochs required for convergence of additional blocks may vary during sequential training and also fine-tuning. 

## References
1. S. Kim and M. Kim, “Bloom-net: Blockwise optimization for masking networks toward scalable andefficient speech enhancement (under review),” in Proc. of the IEEE International Conference on Acoustics,Speech, and Signal Processing (ICASSP), 2021.
