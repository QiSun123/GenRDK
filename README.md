# GenRDK
Code for WWW 2024 paper [Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction.](https://dl.acm.org/doi/10.1145/3589334.3645678)
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
  python CoR.py
```
put the openai api key here in CoR.py
```
  openai.api_key = "private openai api" 
```
We generate 500 documents for each unseen relation type in our experiments. When $m==5$, which means the number of synthetic data is 2500. 
```
  num=500
  relation_types = json.load(open(relation_prompt))
  for i in range(num):
    print("***********",i,"***********")
    generate(relation_types)
```
## Train the downstream models
We follow the [llama-recipes](https://github.com/meta-llama/llama-recipes) to fine-tune (LoRA) the downstream DocRTE model and obtain pseudo labels. 
We follow the format of alpaca and fine-tune the llama2-13b-chat.
```
  python to_alpaca.py 
```
Transfer the DocRED data format to Alpaca format.
```
# put your data_path and save_path.
ToAlpaca_syn(
    origin_path="synthetic_data_m5_s5/train_synthetic_data.json",
    output_path="alpaca_dataset_m5_s5/alpaca_train_synthetic_data.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_unseen_path="meta_m5_s5/rel2id_unseen.json"
)
ToAlpaca_syn(
    origin_path="synthetic_data_m5_s5/train_denoise_data.json",
    output_path="alpaca_dataset_m5_s5/alpaca_train_denoise_data.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_unseen_path="meta_m5_s5/rel2id_unseen.json"
)

ToAlpaca_seen(
    origin_path="dataset_Redocred_m5_s5/train_seen.json",
    output_path="alpaca_dataset_m5_s5/alpaca_train_seen.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_seen_path="meta_m5_s5/seenrel2id_info.json"
)

ToAlpaca_seen(
    origin_path="dataset_Redocred_m5_s5/dev_seen.json",
    output_path="alpaca_dataset_m5_s5/alpaca_dev_seen.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_seen_path="meta_m5_s5/seenrel2id_info.json"
)

ToAlpaca_seen(
    origin_path="dataset_Redocred_m5_s5/test_seen.json",
    output_path="alpaca_dataset_m5_s5/alpaca_test_seen.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_seen_path="meta_m5_s5/seenrel2id_info.json"
)
```
We follow the [UGDRE](https://github.com/QiSun123/UGDRE) to train the downstream DocRE model.
## Evaluation Prompt
We use the same prompt format for different LLMs to obtain relation triplets.
```
### Instruction:\nPerform the document-level relation triplet extraction task. Give you the document and pre-defined relation types, you need to extract possible relation triplets. Provide each relation triplet in the following format: (head entity, tail entity, relation type)\n\n###Input:\n The document:\n{document}\nThe pre-defined relation types: "{relation_types}".\n\n### Response:"""
       
```
## Consistency Guided Knowledge Denoising Strategy
```
  python denoise.py 
```
put your pseudo label path here in denoise.py.
```
denoise(predictResult="", # put your path of pseudo labels here
        rel2id_unseen="meta_m5_s5/rel2id_unseen.json",
        train_unseen="synthetic_data_m5_s5/train_synthetic_data.json",
        savePath="synthetic_data_m5_s5/train_denoise_data.json")
```
The pseudo label format (json list format):
```
[
    {"title": "The Godfather id_522", "h_idx": "The Godfather", "t_idx": "Francis Ford Coppola", "r": "P58"},
] 
```

## Training Procedure
<img width="356" alt="image" src="https://github.com/QiSun123/GenRDK/assets/91941077/debaf93d-8f48-45b7-b3db-331ff9e131ea">

## Citation
```
@inproceedings{10.1145/3589334.3645678,
author = {Sun, Qi and Huang, Kun and Yang, Xiaocui and Tong, Rong and Zhang, Kun and Poria, Soujanya},
title = {Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589334.3645678},
doi = {10.1145/3589334.3645678},
booktitle = {Proceedings of the ACM on Web Conference 2024},
pages = {4407–4416},
numpages = {10},
series = {WWW '24}
}
```

