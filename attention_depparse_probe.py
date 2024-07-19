"""Compute UAS using the argmax attention weight.

Adapted from https://github.com/clarkkev/attention-analysis/blob/master/Syntax_Analysis.ipynb.
"""
import argparse
import collections
import json
import os
import pickle
import numpy as np
from tqdm import tqdm


def load_pickle(fname):
    with open(fname, "rb") as f:
        # add, encoding="latin1") if using python3 and downloaded data
        return pickle.load(f)


# Code for evaluating individual attention maps and baselines


def evaluate_predictor(prediction_fn, dev_data, print_output=False):
    """Compute accuracies for each relation for the given predictor."""
    n_correct, n_incorrect = collections.Counter(), collections.Counter()
    for example in dev_data:
        words = example["words"]
        predictions = prediction_fn(example)
        for i, (p, y, r) in enumerate(
            zip(predictions, example["heads"], example["relns"])
        ):
            is_correct = p == y
            if r == "poss" and p < len(words):
                # Special case for poss (see discussion in Section 4.2)
                if i < len(words) and words[i + 1] == "'s" or words[i + 1] == "s'":
                    is_correct = predictions[i + 1] == y
            if is_correct:
                n_correct[r] += 1
                n_correct["all"] += 1
            else:
                n_incorrect[r] += 1
                n_incorrect["all"] += 1
        out = collections.defaultdict(float)
        for k in n_incorrect.keys():
            out[k] = n_correct[k] / float(n_correct[k] + n_incorrect[k])
    return out


def attn_head_predictor(layer, head, mode="normal"):
    """Assign each word the most-attended-to other word as its head."""

    def predict(example):
        attn = np.array(example["attns"][layer][head])
        if mode == "transpose":
            attn = attn.T
        elif mode == "both":
            attn += attn.T
        else:
            assert mode == "normal"
        # ignore attention to self and [CLS]/[SEP] tokens
        attn[range(attn.shape[0]), range(attn.shape[0])] = 0
        attn = attn[1:-1, 1:-1]
        return np.argmax(attn, axis=-1) + 1  # +1 because ROOT is at index 0

    return predict


def offset_predictor(offset):
    """Simple baseline: assign each word the word a fixed offset from
    it (e.g., the word to its right) as its head."""

    def predict(example):
        return [
            max(0, min(i + offset + 1, len(example["words"])))
            for i in range(len(example["words"]))
        ]

    return predict


def get_scores(data, mode="normal", num_heads=12, num_layers=12):
    """Get the accuracies of every attention head."""
    scores = collections.defaultdict(dict)
    for layer in range(num_layers):
        for head in range(num_heads):
            scores[layer][head] = evaluate_predictor(
                attn_head_predictor(layer, head, mode), data, print_output=True
            )
    return scores


def get_all_scores(reln, attn_head_scores):
    """Get all attention head scores for a particular relation."""
    all_scores = []
    for key, layer_head_scores in attn_head_scores.items():
        for layer, head_scores in layer_head_scores.items():
            for head, scores in head_scores.items():
                all_scores.append((scores[reln], layer, head, key))  # here
    return sorted(all_scores, reverse=True)


def get_uas(data, outfile, num_heads=12, num_layers=12):
    attn_head_scores = {
        "dep->head": get_scores(
            data, "normal", num_heads=num_heads, num_layers=num_layers
        ),
        "head<-dep": get_scores(
            data, "transpose", num_heads=num_heads, num_layers=num_layers
        ),
    }
    baseline_scores = {
        i: evaluate_predictor(offset_predictor(i), data) for i in range(-3, 3)
    }

    # Find the most common relations in our data
    reln_counts = collections.Counter()
    for example in data:
        for reln in example["relns"]:
            reln_counts[reln] += 1

    # Compare the best attention head to baselines across the most common relations.
    # This produces the scores in Table 1
    with open(outfile, "w") as f:
        for row, (reln, _) in enumerate([("all", 0)] + reln_counts.most_common()):
            if reln == "root" or reln == "punct":
                continue
            if reln_counts[reln] < 100 and reln != "all":
                break

            uas, layer, head, direction = sorted(
                s for s in get_all_scores(reln, attn_head_scores)
            )[-1]
            baseline_uas, baseline_offset = max(
                (scores[reln], i) for i, scores in baseline_scores.items()
            )
            f.write(
                "{:8s},{:5d},{:.1f},offset={:2d},{:.1f},{:}-{:},{:}\n".format(
                    reln[:8],
                    reln_counts[reln],
                    100 * uas,
                    baseline_offset,
                    100 * baseline_uas,
                    layer,
                    head,
                    direction,
                )
            )


def output_uas_for_dir(args, model_dir):
    try:
        data = load_pickle(f"{model_dir}/{args.attn_map_filename}")
    except Exception as e:
        raise Exception(
            f"Error while loading {model_dir}/{args.attn_map_filename}: {e}"
        )
    outfile = f"{model_dir}/{args.output_filename}"
    print(f"Getting UAS for {model_dir}...")
    config = json.load(open(f"{model_dir}/config.json"))
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    get_uas(data, outfile, num_heads=num_heads, num_layers=num_layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bert-parent-dir",
        default=None,
        help="Directory containing all BERT checkpoint directories.",
    )
    parser.add_argument("--attn_map_filename", default="dev_attn.pkl")
    parser.add_argument("--output_filename", default="depparse_uas.csv")
    args = parser.parse_args()

    # Check parent dir for attention map dump first.
    if os.path.exists(
        f"{args.bert_parent_dir}/{args.attn_map_filename}"
    ) and not os.path.exists(f"{args.bert_parent_dir}/{args.output_filename}"):
        output_uas_for_dir(args, args.bert_parent_dir)

    for path in tqdm(os.listdir(args.bert_parent_dir)):
        model_dir = os.path.join(args.bert_parent_dir, path)
        if os.path.exists(
            f"{model_dir}/{args.attn_map_filename}"
        ) and not os.path.exists(f"{model_dir}/{args.output_filename}"):
            output_uas_for_dir(args, model_dir)


if __name__ == "__main__":
    main()
