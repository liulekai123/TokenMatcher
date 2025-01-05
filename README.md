# TokenMatcher: Diverse Tokens Matching for Unsupervised Visible-Infrared Person Re-Identification
<img src="https://raw.githubusercontent.com/liulekai123/TokenMatcher/main/examples/pipeline.png" />
This is a code implementation and supplement to TokenMatcher: Diverse Tokens Matching for Unsupervised Visible-Infrared Person Re-Identification, which was accepted by AAAI 2025.

# Contributions
1. We introduce a TokenMatcher, a novel framework designed to extract reliable cross-modality fine-grained person features, facilitating accurate cross-modality correspondences.
2. We present the Diverse Tokens Neighbor Learning (DTNL) module, which identifies reliable neighbors. This capability allows the model to effectively capture modality-invariant and discriminative features.
3. We propose the Homogeneous Fusion (HF) module, which aims to minimize the differences between various camera views, thereby drawing clusters with the same identity closer together.
4. Experiments on SYSU-MM01 and RegDB datasets demonstrate the superiority of our method compared with existing US-VI-ReID methods.

# Prepare Datasets
Put SYSU-MM01 and RegDB dataset (run prepare_sysu.py and prepare_regdb.py to convert to market1501 format) into data/sysu and data/regdb. (Following previous work [ADCA](https://github.com/yangbincv/ADCA))

# Prepare Pre-trained model
Following [SDCL](https://github.com/yangbincv/SDCL), we adopt the self-supervised pre-trained models (ViT-B/16+ICS) from [Self-Supervised Pre-Training for Transformer-Based Person Re-Identification](https://github.com/damo-cv/TransReID-SSL?tab=readme-ov-file)

# Training
1. sh run_train_sysu.sh
2. sh run_train_regdb.sh

# Models
| Datestes                   | Rank-1 | mAP  | Download                                                          |
|----------------------------|--------|------|-------------------------------------------------------------------|
| SYSU-MM01 (All Search)     | 65.07% |62.79%| [model](https://pan.baidu.com/s/12NLKGZIJ3urq8Q8gGqrJyQ?pwd=0000) |
| RegDB (Visible to Infrared)| 92.96% |86.32%| [model](https://pan.baidu.com/s/1K9KCIbJRXxRms6J8o_rR3Q?pwd=0000) |


# Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[SDCL](https://github.com/yangbincv/SDCL). [ADCA](https://github.com/yangbincv/ADCA). [TransReID](https://github.com/damo-cv/TransReID-SSL?tab=readme-ov-file). [DC-former](https://github.com/ant-research/Diverse-and-Compact-Transformer). [CALR](https://github.com/leeBooMla/CALR)
