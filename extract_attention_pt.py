"""
Extract attention maps from either a pre-trained model on the HF hub or saved locally
on a given input dataset.
"""

import argparse
import json
import logging
import math
import numpy as np
import os
import pickle
import six
import torch
import unicodedata

from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorWithPadding


def write_pickle(o, path):
    if "/" in path:
        parent_dir = path.rsplit("/", 1)[0]
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    with open(path, "wb") as f:
        pickle.dump(o, f, -1)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return ["".join(x) for x in output]


def tokenize_and_align(tokenizer, words, cased, max_seq_len=128):
    """Given already-tokenized text (as a list of strings), returns a list of
    lists where each sub-list contains BERT-tokenized tokens for the
    correponding word.

    Args:
        tokenizer: A BertTokenizer
        words: A list of already-tokenized tokens.
        cased: True or false whether to use case-sensitive logic
    """
    words = ["[CLS]"] + words + ["[SEP]"]
    tokenized_words = []
    for word in words:
        word = convert_to_unicode(word)
        word = clean_text(word)
        if word == "[CLS]" or word == "[SEP]":
            word_toks = [word]
        else:
            if not cased:
                word = word.lower()
                word = run_strip_accents(word)
            word_toks = run_split_on_punc(word)

        tokenized_word = []
        for word_tok in word_toks:
            tokenized_word += tokenizer.tokenize(word_tok)
        tokenized_words.append(tokenized_word)

    i = 0
    word_to_tokens = []
    for word in tokenized_words:
        tokens = []
        for _ in word:
            tokens.append(i)
            i += 1
            if i >= max_seq_len:
                break
        word_to_tokens.append(tokens)
        if i >= max_seq_len:
            break
    # assert len(word_to_tokens) == len(words)

    return word_to_tokens


def get_word_word_attention(token_token_attention, words_to_tokens, mode="first"):
    """Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings)."""

    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
        try:
            word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
        except IndexError as e:
            logging.error(
                f"words_to_tokens: {words_to_tokens}\nword_word_attention dimensions: {word_word_attention.shape}\nword[0]: {word[0]}\nword: {word}"
            )
            raise e
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

    # several options for combining attention maps for words that have been split
    # we use "mean" in the paper
    for word in words_to_tokens:
        if mode == "first":
            pass
        elif mode == "mean":
            word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
        elif mode == "max":
            word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
            word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
        else:
            raise ValueError("Unknown aggregation mode", mode)
    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

    return word_word_attention


def make_attn_word_level(data, tokenizer, cased):
    for features in tqdm(data):
        words_to_tokens = tokenize_and_align(
            tokenizer,
            features["words"],
            cased,
            max_seq_len=len(features["attns"][0][0][0]),
        )
        # assert sum(len(word) for word in words_to_tokens) == len(features["tokens"])
        features["attns"] = np.stack(
            [
                [
                    get_word_word_attention(attn_head, words_to_tokens, mode="mean")
                    for attn_head in layer_attns
                ]
                for layer_attns in features["attns"]
            ]
        )


def extract_attention(args, model, tokenizer, tokenized_data):
    output = []
    tokenized_data.map(
        lambda row: output.append(
            {"words": row["words"], "relns": row["relns"], "heads": row["heads"]}
        )
    )
    with torch.no_grad():
        num_batches = math.ceil(len(tokenized_data) / 16)
        for i in tqdm(range(num_batches), desc="Computing attentions..."):
            input_ids = torch.tensor(
                tokenized_data["input_ids"][i * 16 : (i + 1) * 16]
            ).cuda()
            attention_mask = torch.tensor(
                tokenized_data["attention_mask"][i * 16 : (i + 1) * 16]
            ).cuda()
            attns = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            ).attentions
            # the batch size of this particular batch, which is <= args.batch_size
            batch_size = attns[0].shape[0]
            for j in tqdm(range(batch_size), desc="Formatting output for batch..."):
                data_idx = i * 16 + j
                text = " ".join(output[data_idx]["words"])
                tokens = tokenizer.tokenize(text)
                output[data_idx]["tokens"] = ["[CLS]"] + tokens + ["[SEP]"]
                seq_len = len(tokens) + 2
                example_attn = torch.stack([layer_attn[j] for layer_attn in attns])[
                    :, :, :seq_len, :seq_len
                ].cpu()
                output[data_idx]["attns"] = example_attn

    if args.word_level:
        print("Converting to word-level attention...")
        make_attn_word_level(output, tokenizer, args.cased)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name of pre-trained HuggingFace model to use. If populated, overrides --model-dir.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
    )
    parser.add_argument(
        "--preprocessed-data-file",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If None, uses the same directory that the model is stored in.",
    )
    parser.add_argument("--output-filename", default="dev_attn_wsj2.pkl")
    parser.add_argument(
        "--word_level",
        default=False,
        action="store_true",
        help="Get word-level rather than token-level attention.",
    )
    parser.add_argument(
        "--cased", default=False, action="store_true", help="Don't lowercase the input."
    )
    parser.add_argument("--batch-size", default=16, type=int)
    args = parser.parse_args()

    print(f"Is GPU available? {torch.cuda.is_available()}")

    if args.model_name is not None:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_dataset("json", data_files={"train": args.preprocessed_data_file})[
        "train"
    ]
    data_w_text = data.map(
        lambda row: {
            "words": row["words"],
            "relns": row["relns"],
            "heads": row["heads"],
            "text": " ".join(row["words"]),
        }
    )
    tokenized = data_w_text.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", max_length=128, truncation=True
        ),
        batched=True,
    )

    if args.model_name is not None:
        print(f"Extracting attention from {args.model_name} pre-trained model.")
        model = BertForMaskedLM.from_pretrained(args.model_name).cuda()
        output_path = os.path.join(args.output_dir, args.output_filename)
        model.config.save_pretrained(args.output_dir)
        output = extract_attention(args, model, tokenizer, tokenized)
        write_pickle(output, output_path)
    elif os.path.exists(os.path.join(args.model_dir, "config.json")):
        output_path = os.path.join(args.model_dir, args.output_filename)
        if os.path.exists(output_path):
            print(f"{output_path} already exists. Continuing...")
            return
        if os.path.exists(
            os.path.join(output_path, "depparse_uas_wsj_pt.csv")
        ) and os.path.exists(os.path.join(output_path, "depparse_uas_scores.csv")):
            print(
                f"{output_path}/depparse_uas_scores.csv and {output_path}/depparse_uas_wsj_pt.csv already exist. Continuing..."
            )
            return
        print(f"Extracting attention for {args.model_dir}...")
        model = BertForMaskedLM.from_pretrained(args.model_dir).cuda()
        output = extract_attention(args, model, tokenizer, tokenized)
        write_pickle(output, output_path)
    else:
        # Load model from subdirectories of --model-dir instead.
        for path in os.listdir(args.model_dir):
            ckpt_dir = os.path.join(args.model_dir, path)
            if os.path.isdir(ckpt_dir) and os.path.exists(
                os.path.join(ckpt_dir, "config.json")
            ):
                output_path = os.path.join(ckpt_dir, args.output_filename)
                if os.path.exists(output_path):
                    print(f"{output_path} already exists. Continuing...")
                    continue
                if os.path.exists(
                    os.path.join(output_path, "depparse_uas_wsj_pt.csv")
                ) and os.path.exists(
                    os.path.join(output_path, "depparse_uas_scores.csv")
                ):
                    print(
                        f"{output_path}/depparse_uas_scores.csv and {output_path}/depparse_uas_wsj_pt.csv already exist. Continuing..."
                    )
                    continue

                print(f"Extracting attention for {ckpt_dir}...")
                model = BertForMaskedLM.from_pretrained(ckpt_dir).cuda()
                output = extract_attention(args, model, tokenizer, tokenized)
                write_pickle(output, output_path)


if __name__ == "__main__":
    main()
