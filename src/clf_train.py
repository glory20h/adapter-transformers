import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from random import randint
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_metric, load_from_disk

import transformers
from transformers import (
    Wav2Vec2Config,
    HubertConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2AdapterModel,
    HubertAdapterModel,
    Trainer,
    AdapterTrainer,
    TrainerCallback,
    TrainingArguments,
    PrefixTuningConfig,
    PfeifferConfig,
    HoulsbyConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger("transformers")

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

# ========================================== CONFIG ==========================================
# MODEL_NAME = "facebook/hubert-large-ll60k"
MODEL_NAME = "facebook/hubert-base-ls960"
DATASET = "superb"
DATASET_CONFIG = "si"
DATA_DIR = "./dataset/VoxCeleb1"
TRAIN_SPLIT_NAME = "train"
EVAL_SPLIT_NAME = "validation"
TEST_SPLIT_NAME = "test"
RESUME_TRAINING = False
EPOCHS = 150
# -1 to disable
MAX_STEPS = 20000
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 1
PER_DEVICE_EVAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE
LEARNING_RATE = 3e-5
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
LR_SCHEDULER_TYPE = "linear"
WARMUP_STEPS = 500
LOGGING_STEPS = 100
EVAL_STEPS = 400
GRADIENT_CHECKPOINTING = True
MAX_LENGTH_SECONDS = 20.0
PREPROCESSING_NUM_WORKERS = None
SET_SEED = False
FINAL_DROPOUT = 0.1
# 'prefix', 'houlsby', 'pfeiffer', None
ADAPTER = None
USE_WEIGHTED_LAYER_SUM = False
# prefix tuning config
PREFIX_LENGTH = 50
BOTTLENECK_SIZE = 512
PREFIX_DROPOUT = 0.05
# adapter config
REDUCTION_FACTOR = 4
LN_AFTER = True
NON_LINEARITY = "gelu"  # Pfeiffer default: "relu", Houlsby default: "swish"
# ========================================== CONFIG ==========================================

OUTPUT_DIR = "./results/" + MODEL_NAME.split("/")[-1] + "-" + DATASET + "-" + DATASET_CONFIG
if ADAPTER == "prefix":
    OUTPUT_DIR = OUTPUT_DIR + "-" + ADAPTER + "-" + str(PREFIX_LENGTH)
elif ADAPTER in ["houlsby", "pfeiffer"]:
    OUTPUT_DIR = OUTPUT_DIR + "-" + ADAPTER + "-" + str(REDUCTION_FACTOR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
    
GRADIENT_CHECKPOINTING = False if ADAPTER == 'prefix' else GRADIENT_CHECKPOINTING
USE_WEIGHTED_LAYER_SUM = True if ADAPTER else USE_WEIGHTED_LAYER_SUM


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to manual data containing the training audio paths and labels."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to "
            "'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the test data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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
    eval_metrics: List[str] = list_field(
        default=["acc"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'acc'`"},
    )
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )


def main():

    # Setup logging
    logging_path = os.path.join(OUTPUT_DIR, 'train.log')
    i = 2
    while os.path.isfile(logging_path) and not RESUME_TRAINING:
        logging_path = os.path.join(OUTPUT_DIR, 'train.log')
        logging_path = logging_path[:-4] + str(i) + logging_path[-4:]
        i += 1
        
    fileHandler = logging.FileHandler(logging_path)
    # streamHandler = logging.StreamHandler(sys.stdout)

    logger.addHandler(fileHandler)
    # logger.addHandler(streamHandler)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    # Log all args
    logger.info("MODEL_NAME = " + str(MODEL_NAME))
    logger.info("DATASET = " + str(DATASET))
    logger.info("DATASET_CONFIG = " + str(DATASET_CONFIG))
    logger.info("TRAIN_SPLIT_NAME = " + str(TRAIN_SPLIT_NAME))
    logger.info("EVAL_SPLIT_NAME = " + str(EVAL_SPLIT_NAME))
    logger.info("TEST_SPLIT_NAME = " + str(TEST_SPLIT_NAME))
    logger.info("RESUME_TRAINING = " + str(RESUME_TRAINING))
    logger.info("MAX_STEPS = " + str(MAX_STEPS))
    logger.info("PER_DEVICE_BATCH_SIZE = " + str(PER_DEVICE_BATCH_SIZE))
    logger.info("GRADIENT_ACCUMULATION = " + str(GRADIENT_ACCUMULATION))
    logger.info("PER_DEVICE_EVAL_BATCH_SIZE = " + str(PER_DEVICE_EVAL_BATCH_SIZE))
    logger.info("LEARNING_RATE = " + str(LEARNING_RATE))
    logger.info("LR_SCHEDULER_TYPE = " + str(LR_SCHEDULER_TYPE))
    logger.info("WARMUP_STEPS = " + str(WARMUP_STEPS))
    logger.info("LOGGING_STEPS = " + str(LOGGING_STEPS))
    logger.info("EVAL_STEPS = " + str(EVAL_STEPS))
    logger.info("GRADIENT_CHECKPOINTING = " + str(GRADIENT_CHECKPOINTING))
    logger.info("MAX_LENGTH_SECONDS = " + str(MAX_LENGTH_SECONDS))
    logger.info("PREPROCESSING_NUM_WORKERS = " + str(PREPROCESSING_NUM_WORKERS))
    logger.info("SET_SEED = " + str(SET_SEED))
    logger.info("FINAL_DROPOUT = " + str(FINAL_DROPOUT))
    logger.info("ADAPTER = " + str(ADAPTER))
    logger.info("USE_WEIGHTED_LAYER_SUM = " + str(USE_WEIGHTED_LAYER_SUM))

    if ADAPTER == "prefix":
        logger.info("PREFIX_LENGTH = " + str(PREFIX_LENGTH))
        logger.info("BOTTLENECK_SIZE = " + str(BOTTLENECK_SIZE))
        logger.info("PREFIX_DROPOUT = " + str(PREFIX_DROPOUT))
    elif ADAPTER:
        logger.info("LN_AFTER = " + str(LN_AFTER))
        logger.info("REDUCTION_FACTOR = " + str(REDUCTION_FACTOR))
        logger.info("NON_LINEARITY = " + str(NON_LINEARITY))

    logger.info("OUTPUT_DIR = " + str(OUTPUT_DIR))

    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME,
        attention_mask=False,           # -> ?
        freeze_feature_encoder=True,
    )

    data_args = DataTrainingArguments(
        dataset_name=DATASET,
        dataset_config_name=DATASET_CONFIG,
        train_split_name=TRAIN_SPLIT_NAME,
        eval_split_name=EVAL_SPLIT_NAME,
        test_split_name=TEST_SPLIT_NAME,
        preprocessing_num_workers=PREPROCESSING_NUM_WORKERS,
        eval_metrics=["accuracy"],
        max_length_seconds=MAX_LENGTH_SECONDS,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False if RESUME_TRAINING else True,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=WARMUP_STEPS,
        dataloader_num_workers=4,
        evaluation_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=1,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
    )

    if SET_SEED:
        set_seed(training_args.seed)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=data_args.use_auth_token,
    )

    # Prepare Data
    processed_data_path = os.path.join("./dataset", DATASET + "-" + DATASET_CONFIG)

    try:
        vectorized_datasets = load_from_disk(processed_data_path)
    except:
        raw_datasets = DatasetDict()

        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            split=data_args.train_split_name,
            use_auth_token=data_args.use_auth_token,
        )

        if data_args.audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.label_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            split=data_args.eval_split_name,
            use_auth_token=data_args.use_auth_token,
        )

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

        raw_datasets["test"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            split=data_args.test_split_name,
            use_auth_token=data_args.use_auth_token,
        )
        
        vectorized_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )
        
        os.makedirs(processed_data_path, exist_ok=True)
        vectorized_datasets.save_to_disk(processed_data_path)

    labels = vectorized_datasets["train"].features[data_args.label_column_name].names

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
        """Randomly sample chunks of 'max_length' seconds from the input audio"""
        sample_length = int(round(sample_rate * max_length))
        if len(wav) <= sample_length:
            return wav
        random_offset = randint(0, len(wav) - sample_length - 1)
        return wav[random_offset : random_offset + sample_length]

    # Apply train_transforms across a batch.
    def train_transforms(batch):
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column_name]:
            wav = random_subsample(
                audio['array'], max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
            )
            output_batch['input_values'].append(wav)
        output_batch['labels'] = [label for label in batch[data_args.label_column_name]]
                
        return output_batch

    def val_transforms(batch):
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column_name]:
            wav = audio['array']
            output_batch['input_values'].append(wav)
        output_batch['labels'] = [label for label in batch[data_args.label_column_name]]
        
        return output_batch

    # Set transforms for train
    vectorized_datasets["train"].set_transform(train_transforms, output_all_columns=False)

    # Set transforms for validation
    vectorized_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    # Set transforms for test
    vectorized_datasets["test"].set_transform(val_transforms, output_all_columns=False)

    config_cls = HubertConfig if 'hubert' in MODEL_NAME else Wav2Vec2Config
    config = config_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=data_args.use_auth_token,
    )

    config.update(
        {
            # "feat_proj_dropout": model_args.feat_proj_dropout,
            # "attention_dropout": model_args.attention_dropout,
            # "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": FINAL_DROPOUT,
            # "mask_time_prob": model_args.mask_time_prob,
            # "mask_time_length": model_args.mask_time_length,
            # "mask_feature_prob": model_args.mask_feature_prob,
            # "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            # "layerdrop": model_args.layerdrop,
            # "activation_dropout": model_args.activation_dropout,
            "use_weighted_layer_sum": USE_WEIGHTED_LAYER_SUM,
        }
    )

    model_cls = HubertAdapterModel if 'hubert' in MODEL_NAME else Wav2Vec2AdapterModel
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        revision=model_args.model_revision,
        use_auth_token=data_args.use_auth_token,
    )

    total_params = sum(p.numel() for p in model.parameters())

    if ADAPTER:
        if ADAPTER == 'prefix':
            adapter_config = PrefixTuningConfig(
                flat=False,
                prefix_length=PREFIX_LENGTH,
                bottleneck_size=BOTTLENECK_SIZE,
                non_linearity="tanh",
                dropout=PREFIX_DROPOUT,
            )
        elif ADAPTER == 'houlsby':
            adapter_config = HoulsbyConfig(
                ln_after=LN_AFTER,
                non_linearity=NON_LINEARITY,
                reduction_factor=REDUCTION_FACTOR,
            )
        elif ADAPTER == 'pfeiffer':
            adapter_config = PfeifferConfig(
                non_linearity=NON_LINEARITY,
                reduction_factor=REDUCTION_FACTOR,
            )
        else:
            raise ValueError("No corresponding adapter config.")
        model.add_adapter(DATASET_CONFIG + "-clf", config=adapter_config, overwrite_ok=True)
        model.train_adapter(DATASET_CONFIG + "-clf")
    else:
        if USE_WEIGHTED_LAYER_SUM:
            model.freeze_base_model()
        else:
            model.freeze_feature_encoder()

    model.add_classification_head(
        head_name=DATASET_CONFIG+"-clf",
        num_labels=len(id2label),
        id2label=id2label,
        overwrite_ok=True,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info("Ratio of trainable parameters: {:.3f}%".format(trainable_params / total_params * 100))

    if ADAPTER == "prefix":
        import copy
        pool = model.base_model.prefix_tuning.prefix_tunings['asr-ctc']
        pool_clone = copy.deepcopy(pool)
        pool_clone.eject()
        prefix_params = sum(p.numel() for p in pool_clone.parameters() if p.requires_grad)
        del pool_clone

        head_params = sum(p.numel() for p in model.heads.parameters() if p.requires_grad)

        trainable_params = head_params + prefix_params
        logger.info("[Prefix] Ratio of final added parameters: {:.3f}%".format(trainable_params / total_params * 100))

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                logger.info(str(logs))

    trainer_cls = AdapterTrainer if ADAPTER else Trainer
    trainer = trainer_cls(
        model=model,
        data_collator=None,  # Use default data collator
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["eval"],
        tokenizer=feature_extractor,
        callbacks=[LogCallback],
    )

    checkpoint = None
    if RESUME_TRAINING:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            checkpoint = last_checkpoint
        else:
            logger.info(
                f"RESUME_TRAINING set to True but checkpoint not detected, training from scratch."
            )
            checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    test_results = trainer.evaluate(vectorized_datasets["test"])
    trainer.log_metrics("eval", test_results)
    trainer.save_metrics("eval", test_results)

if __name__ == "__main__":
    main()