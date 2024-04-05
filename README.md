# GenRDK
Code for WWW 2024 paper [Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction.](https://arxiv.org/abs/2401.13598)
## Dataset
We perform experiments on [DocRED](https://github.com/thunlp/DocRED) and [RE-DocRED](https://github.com/tonytan48/re-docred).
## Settings
We follow the zero-shot setting of Chia et, al. that partitions the pre-defined relation types into a seen relation set and an unseen relation set. Documents with labels of the seen set are available for training while documents that contain the unseen set are used for evaluation. 

The unseen relations are randomly selected from the relation types in datasets. For a fair comparison, we evaluate models under different sizes $m\in\{5,10\}$ of unseen relation sets and randomly sample three times for each size to obtain different unseen relation sets. 
### Check the partitions
```
python check.py
```
We present one demo in meta_m5_s5, which contains 5 different unseen relation types.
## Obtain synthetic data by ChatGPT
```
  python CoR.py # put your openai api key in CoR.py
```
## Train the downstream models
We follow the [llama-recipes](https://github.com/meta-llama/llama-recipes) to fine-tune (LoRA) the downstream DocRTE model.

We follow the [UGDRE](https://github.com/QiSun123/UGDRE) to train the downstream DocRE model.
## Citation
```
@article{sun2024Consistency,
  title={Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction},
  author={Qi Sun, Kun Huang, Xiaocui Yang, Rong Tong, Kun Zhang, Soujanya Poria},
  journal={Proceedings of WWW},
  year={2024}
}
```

