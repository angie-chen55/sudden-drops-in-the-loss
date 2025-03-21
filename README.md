# Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs

This repository provides the code for the paper "Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs," by Angelica Chen, Ravid Shwartz-Ziv, Kyunghyun Cho, Matthew L Leavitt, and Naomi Saphra.

Our datasets have been uploaded to HF Hub:
- Tokenized pre-training dataset: https://huggingface.co/datasets/angie-chen55/bert_pretraining_data (This contains both BookCorpus and Wikipedia, but only the Wikipedia portion of the dataset has "heads" and "relns" labelled, so these are set to -1's for the other examples)
- Parses of 1K sample of WSJ data: https://huggingface.co/datasets/angie-chen55/wsj-dep-parses-1k

We have also released all checkpoints of the 3 seeds of BERT-Base that we trained: https://huggingface.co/models?search=angie-chen55/bert-base-seed
For the first 15K steps, we have checkpoints every 1K steps. Afterwards, checkpoint intervals range between 5K, 10K, and 50K steps.

# SAS-regularized training
To run MLM training with SAS regularization with DDP:
```
python -m torch.distributed.launch --nproc_per_node=4 run_mlm_reg_depparse.py \
    --config_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --dataset_name angie-chen55/bert_pretraining_data \
    --do_train \
    --warmup_steps=10000 \
    --save_steps=5000 \
    --max_steps=500000 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --lambda_reg=0.001 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --max_seq_length=512 \
    --fp16 \
    --reg_type max \
    --preprocessing_num_workers 4 \
    --reg_type=max --seed=0 --output_dir=<INSERT_OUTPUT_DIR>
```


## Citation
To cite our work, please use the below citation:
```
@inproceedings{
chen2024sudden,
title={Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in {MLM}s},
author={Angelica Chen and Ravid Shwartz-Ziv and Kyunghyun Cho and Matthew L Leavitt and Naomi Saphra},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=MO5PiKHELW}
}
```
