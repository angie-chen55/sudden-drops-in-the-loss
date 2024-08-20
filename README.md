# Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs

This repository provides the code for the paper "Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs," by Angelica Chen, Ravid Shwartz-Ziv, Kyunghyun Cho, Matthew L Leavitt, and Naomi Saphra.

Our datasets have been uploaded to HF Hub:
- Tokenized pre-training dataset: https://huggingface.co/datasets/angie-chen55/bert_pretraining_data (This contains both BookCorpus and Wikipedia, but only the Wikipedia portion of the dataset has "heads" and "relns" labelled, so these are set to -1's for the other examples)
- Parses of 1K sample of WSJ data: https://huggingface.co/datasets/angie-chen55/wsj-dep-parses-1k


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