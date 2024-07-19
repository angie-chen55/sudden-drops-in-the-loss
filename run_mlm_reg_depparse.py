"""
Fine-tuning a MLM model, with syntactic regularization.
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
from datasets import concatenate_datasets, load_dataset, Features, Sequence, Value

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    lambda_reg: float = field(default=0.01)
    reg_type: Optional[str] = field(
        default="uniform",
        metadata={
            "help": "The type of regularization to apply - uniform, max, or max_margin."
        },
    )
    reg_only_head: bool = field(
        default=False,
        metadata={
            "help": "Whether to regularize only attention on the dependency relation heads."
        },
    )
    reg_only_children: bool = field(
        default=False,
        metadata={
            "help": "Whether to regularize only attention on the dependency relation children."
        },
    )
    uniform_reg_p_norm: float = field(
        default=1,
        metadata={
            "help": "If uniformly regularizing, which p-norm to use. Default p=1."
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    parallelize: bool = field(
        default=False,
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library). (CUSTOM: Use 'bert_dataset' to combine Wikipedia and BookCorpus.)"
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )


class TrainerWithAttentionDepParseReg(Trainer):
    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        lambda_reg: float = 0.01,
        reg_heads=True,
        reg_children=True,
        reg_type="uniform",
        uniform_reg_p_norm: float = 1.0,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.lambda_reg = lambda_reg
        self.reg_heads = reg_heads
        self.reg_children = reg_children
        self.reg_type = reg_type
        self.uniform_reg_p_norm = uniform_reg_p_norm

    def get_uniform_reg_val(self, attns, example_heads, num_layers, num_heads, row_idx):
        ex_reg = 0.0
        if len(example_heads) > 511:
            example_heads = example_heads[:511]
        if self.reg_heads:
            ex_reg += torch.sum(
                attns[
                    :,
                    row_idx,
                    :,
                    torch.LongTensor(range(1, len(example_heads) + 1)),
                    example_heads,
                ]
            )
        if self.reg_children:
            ex_reg += torch.sum(
                attns[
                    :,
                    row_idx,
                    :,
                    example_heads,
                    torch.LongTensor(range(1, len(example_heads) + 1)),
                ]
            )
        return ex_reg

    def get_batch_uniform_reg_val(self, attns, batch_reln_heads):
        attns = torch.float_power(attns, self.uniform_reg_p_norm)
        sum_attns = torch.sum(torch.sum(attns, dim=2), dim=0)
        batch_size = sum_attns.shape[0]
        batch_reln_heads = batch_reln_heads[:, :511]
        num_tokens = batch_reln_heads.shape[1]

        # padding of batch_reln_heads is -1, so add last row and column of zeros for each example in the batch
        zero_padding_right = torch.zeros((batch_size, sum_attns.shape[1], 1)).cuda()
        sum_attns = torch.cat((sum_attns, zero_padding_right), 2)
        zero_padding_bottom = torch.zeros((batch_size, 1, sum_attns.shape[2])).cuda()
        sum_attns = torch.cat((sum_attns, zero_padding_bottom), 1)
        xx = torch.LongTensor(range(0, batch_size)).repeat_interleave(num_tokens)
        yy = torch.LongTensor(range(1, num_tokens + 1)).repeat(batch_size)
        zz = torch.flatten(batch_reln_heads)
        ex_reg = 0.0
        try:
            if self.reg_heads:
                ex_reg += torch.sum(sum_attns[xx, yy, zz]) ** (
                    1.0 / self.uniform_reg_p_norm
                )
            if self.reg_children:
                ex_reg += torch.sum(sum_attns[xx, zz, yy]) ** (
                    1.0 / self.uniform_reg_p_norm
                )
        except Exception as e:
            print(f"num tokens: {num_tokens}")
            print(f"batch size: {batch_size}")
            print(f"z: {zz}")
            print(f"sum_attns shape: {sum_attns.shape}")
            raise e
        return ex_reg

    def get_batch_max_reg_val(self, attns, batch_reln_heads):
        # attns has dimension [num_layers, batch_size, num_attn_heads, max_seq_length, max_seq_length]
        # Get the max attn output across layers and heads, per example per token pair
        max_attns = torch.max(torch.max(attns, 2).values, 0).values
        batch_size = max_attns.shape[0]
        batch_reln_heads = batch_reln_heads[:, :511]
        num_tokens = batch_reln_heads.shape[1]

        # padding of batch_reln_heads is -1, so add last row and column of zeros for each example in the batch
        zero_padding_right = torch.zeros((batch_size, max_attns.shape[1], 1)).cuda()
        max_attns = torch.cat((max_attns, zero_padding_right), 2)
        zero_padding_bottom = torch.zeros((batch_size, 1, max_attns.shape[2])).cuda()
        max_attns = torch.cat((max_attns, zero_padding_bottom), 1)
        xx = torch.LongTensor(range(0, batch_size)).repeat_interleave(num_tokens)
        yy = torch.LongTensor(range(1, num_tokens + 1)).repeat(batch_size)
        zz = torch.flatten(batch_reln_heads)
        ex_reg = 0.0
        try:
            if self.reg_heads:
                ex_reg += torch.sum(max_attns[xx, yy, zz])
            if self.reg_children:
                ex_reg += torch.sum(max_attns[xx, zz, yy])
        except Exception as e:
            print(f"num tokens: {num_tokens}")
            print(f"batch size: {batch_size}")
            print(f"z: {zz}")
            print(f"max_attns shape: {max_attns.shape}")
            raise e
        return ex_reg

    def get_max_reg_val(self, attns, example_heads, num_layers, row_idx):
        ex_reg = 0.0
        if self.reg_heads:
            reln_attn = attns[:, row_idx, :, range(len(example_heads)), example_heads]
            reln_attn_max = torch.max(reln_attn, dim=0).values
            reln_attn_max = torch.max(reln_attn_max, dim=0).values
            ex_reg += torch.sum(reln_attn_max)
        if self.reg_children:
            reln_attn = attns[:, row_idx, :, example_heads, range(len(example_heads))]
            reln_attn_max = torch.max(reln_attn, dim=0).values
            reln_attn_max = torch.max(reln_attn_max, dim=0).values
            ex_reg += torch.sum(reln_attn_max)
        return ex_reg / len(example_heads)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ):
        """
        Expects inputs to include "heads" and "relns" keys too.
        """
        labels = inputs["labels"].cuda()
        outputs = model(
            input_ids=inputs["input_ids"].cuda(),
            token_type_ids=inputs["token_type_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            labels=labels,
            output_attentions=True,
            return_dict=True,
        )
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Now add dependency parsing regularization - suppresses attention weights corresponding to the head
        # of each dependency relation
        attns = outputs.attentions
        heads = inputs["heads"]
        relns = inputs["relns"]
        # Get rows in batch that have "heads" and "relns" populated

        attns = torch.stack(attns)

        if self.reg_type == "uniform":
            heads = torch.where(heads > 511, 511, heads)
            loss += self.lambda_reg * self.get_batch_uniform_reg_val(attns, heads)
        else:
            heads = torch.where(heads > 511, 511, heads)
            loss += self.lambda_reg * self.get_batch_max_reg_val(attns, heads)

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        >> IMPORTANT: Overrides the HF LR scheduler so that num_training_steps is always set to 1M, which works
        >> better for multi-stage training jobs.

        Args:
            num_training_steps (int): The number of training steps to do.
            optimizer (torch.optim.Optimizer): Optimizer to wrap in the scheduler.
        """
        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = self._get_linear_schedule_with_warmup(
                opt,
                self.args.get_warmup_steps(num_training_steps),
            )
        return self.lr_scheduler

    def _get_linear_schedule_with_warmup(
        self, optimizer, num_warmup_steps, last_epoch=-1
    ):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer (:class:`~torch.optim.Optimizer`):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        num_training_steps = 10**6

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    assert model_args.reg_type in [
        "uniform",
        "max",
        "max_margin",
    ], "--reg_type must be one of ['uniform', 'max', 'max_margin']."

    # Log on each process the small summary:
    logger.info(f"available devices: {torch.cuda.device_count()}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if model_args.reg_only_head and model_args.reg_only_children:
        raise ValueError(
            "Cannot have both --reg-only-head and --reg-only-children set."
        )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data args {data_args}")
    logger.info(f"Model args {model_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        #         "cache_dir": model_args.cache_dir,
        #         "revision": model_args.model_revision,
        #         "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        # if config.hidden_size % 12 != 0:
        #     new_hs = int(math.ceil(config.hidden_size / 12)) * 12
        #     config.update({"hidden_size": new_hs})
        # config.update({"num_attention_heads": 12, "num_hidden_layers": 12})
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Load already tokenized pretraining data
    raw_datasets = datasets.load_dataset(data_args.dataset_name)

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)
    if model_args.parallelize:
        model.parallelize()
    else:
        model = model.cuda()

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=None,
    )

    # Initialize our Trainer
    training_args.remove_unused_columns = False
    trainer = TrainerWithAttentionDepParseReg(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        lambda_reg=model_args.lambda_reg,
        reg_heads=not model_args.reg_only_children,
        reg_children=not model_args.reg_only_head,
        reg_type=model_args.reg_type,
        uniform_reg_p_norm=model_args.uniform_reg_p_norm,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            # if at step 0, save checkpoint
            ckpt0_dir = os.path.join(training_args.output_dir, "checkpoint-0")
            trainer.save_model(output_dir=ckpt0_dir)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
