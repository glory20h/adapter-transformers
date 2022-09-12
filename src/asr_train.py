import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
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
    ParallelConfig,
    ConfigUnion,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger("transformers")

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

# ========================================== CONFIG ==========================================
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
# MODEL_NAME = "facebook/wav2vec2-large"
# MODEL_NAME = "facebook/hubert-large-ll60k"
# MODEL_NAME = "facebook/hubert-base-ls960"
DATASET = "common_voice"
# DATASET = "mozilla-foundation/common_voice_3_0"
# DATASET = "superb"
DATASET_CONFIG = "tr"
# DATASET_CONFIG = "es"
# DATASET_CONFIG = "asr"
TRAIN_SPLIT_NAME = "train"
EVAL_SPLIT_NAME = "validation"
TEST_SPLIT_NAME = "test"
FORCE_REDOWNLOAD = False
RESUME_TRAINING = False
EPOCHS = 100
# -1 to disable
MAX_STEPS = -1       
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
PER_DEVICE_EVAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE
LEARNING_RATE = 1e-4
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
LR_SCHEDULER_TYPE = "linear"
WARMUP_STEPS = 500
LOGGING_STEPS = 100
EVAL_STEPS = 400
GRADIENT_CHECKPOINTING = True
MAX_DURATION_IN_SECONDS = 20.0
PREPROCESSING_NUM_WORKERS = None
DATALOADER_NUM_WORKERS = 8
SET_SEED = False
FINAL_DROPOUT = 0.1
# 'prefix', 'houlsby', 'pfeiffer', 'parallel', 'union', None
ADAPTER = None
# prefix tuning config
PREFIX_LENGTH = 100
BOTTLENECK_SIZE = 512
PREFIX_DROPOUT = 0.05
# adapter config
REDUCTION_FACTOR = 16
LN_AFTER = False
NON_LINEARITY = "relu"  # pfeiffer, parallel default: "relu", houlsby default: "swish"
# layer weights
USE_WEIGHTED_LAYER_SUM = False
# ========================================== CONFIG ==========================================

OUTPUT_DIR = "./results/" + MODEL_NAME.split("/")[-1] + "-" + DATASET + "-" + DATASET_CONFIG

if ADAPTER == "prefix":
    OUTPUT_DIR = OUTPUT_DIR + "-" + ADAPTER + "-" + str(PREFIX_LENGTH)
elif ADAPTER in ["houlsby", "pfeiffer", "parallel"]:
    OUTPUT_DIR = OUTPUT_DIR + "-" + ADAPTER + "-" + str(REDUCTION_FACTOR)
elif ADAPTER == 'union':
    OUTPUT_DIR = OUTPUT_DIR + "-" + ADAPTER + "-r" + str(REDUCTION_FACTOR) + "-l" + str(PREFIX_LENGTH)

os.makedirs(OUTPUT_DIR, exist_ok=True)
    
GRADIENT_CHECKPOINTING = False if ADAPTER in ['prefix', 'union'] else GRADIENT_CHECKPOINTING
USE_WEIGHTED_LAYER_SUM = True if ADAPTER else USE_WEIGHTED_LAYER_SUM

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
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
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
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
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
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
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
            " passed to the tokenizer for tokenization. Note that"
            " this is only relevant if the model classifies the"
            " input audio to a sequence of phoneme sequences."
        },
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def main():

    # Setup logging
    logging_path = os.path.join(OUTPUT_DIR, 'train.log')
    i = 2
    while os.path.isfile(logging_path) and not RESUME_TRAINING:
        logging_path = os.path.join(OUTPUT_DIR, 'train.log')
        logging_path = logging_path[:-4] + str(i) + logging_path[-4:]
        i += 1
        
    fileHandler = logging.FileHandler(logging_path)

    logger.addHandler(fileHandler)

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
    logger.info("EPOCHS = " + str(EPOCHS))
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
    logger.info("MAX_DURATION_IN_SECONDS = " + str(MAX_DURATION_IN_SECONDS))
    logger.info("PREPROCESSING_NUM_WORKERS = " + str(PREPROCESSING_NUM_WORKERS))
    logger.info("SET_SEED = " + str(SET_SEED))
    logger.info("FINAL_DROPOUT = " + str(FINAL_DROPOUT))
    logger.info("ADAPTER = " + str(ADAPTER))

    if ADAPTER in ["prefix", "union"]:
        logger.info("PREFIX_LENGTH = " + str(PREFIX_LENGTH))
        logger.info("BOTTLENECK_SIZE = " + str(BOTTLENECK_SIZE))
        logger.info("PREFIX_DROPOUT = " + str(PREFIX_DROPOUT))

    if ADAPTER in ["houlsby", "pfeiffer", "parallel", "union"]:
        logger.info("LN_AFTER = " + str(LN_AFTER))
        logger.info("REDUCTION_FACTOR = " + str(REDUCTION_FACTOR))
        logger.info("NON_LINEARITY = " + str(NON_LINEARITY))

    logger.info("OUTPUT_DIR = " + str(OUTPUT_DIR))

    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME,
        layerdrop=0.0,
        freeze_feature_encoder=True,
    )

    data_args = DataTrainingArguments(
        dataset_name=DATASET,
        dataset_config_name=DATASET_CONFIG,
        train_split_name=TRAIN_SPLIT_NAME,
        eval_split_name=EVAL_SPLIT_NAME,
        test_split_name=TEST_SPLIT_NAME,
        text_column_name="sentence" if "common_voice" in DATASET else "text",
        preprocessing_num_workers=PREPROCESSING_NUM_WORKERS,
        chars_to_ignore=[',','?','.','!','-','\;','\:','\"','“','%','‘','”','�'],
        eval_metrics=["wer", "cer"],
        max_duration_in_seconds=MAX_DURATION_IN_SECONDS,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True if not RESUME_TRAINING else False,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=WARMUP_STEPS,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        evaluation_strategy="steps",
        length_column_name="input_length",
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=1,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=True,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    if SET_SEED:
        set_seed(training_args.seed)

    # Prepare Data
    raw_datasets = DatasetDict()

    raw_datasets["train"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.train_split_name,
        use_auth_token=data_args.use_auth_token,
        download_mode="force_redownload" if FORCE_REDOWNLOAD else None,
    )

    if data_args.audio_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if data_args.text_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
                                                                
    raw_datasets["eval"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.eval_split_name,
        use_auth_token=data_args.use_auth_token,
    )

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))
        
    raw_datasets["test"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.test_split_name,
        use_auth_token=data_args.use_auth_token,
    )

    chars_to_ignore_regex = (
        f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
    )
    text_column_name = data_args.text_column_name

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
        else:
            batch["target_text"] = batch[text_column_name].lower() + " "
        return batch

    with training_args.main_process_first(desc="dataset map special characters removal"):
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            remove_columns=[text_column_name],
            desc="remove special characters from datasets",
        )

    config_cls = HubertConfig if 'hubert' in MODEL_NAME else Wav2Vec2Config
    config = config_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=data_args.use_auth_token,
    )

    def create_vocabulary_from_data(
        datasets: DatasetDict,
        word_delimiter_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
    ):
        # Given training and test labels create vocabulary
        def extract_all_chars(batch):
            all_text = " ".join(batch["target_text"])
            vocab = list(set(all_text))
            return {"vocab": [vocab], "all_text": [all_text]}
        
        vocabs = datasets.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=datasets["train"].column_names,
        )
        
        vocab_set = set()
        for i, dataset in enumerate(vocabs.values()):
            vocab_set = vocab_set | set(dataset['vocab'][0])
        
        vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
        
        # replace white space with delimiter token
        if word_delimiter_token is not None:
            vocab_dict[word_delimiter_token] = vocab_dict[" "]
            del vocab_dict[" "]
            
        # add unk and pad token
        if unk_token is not None:
            vocab_dict[unk_token] = len(vocab_dict)
        
        if pad_token is not None:
            vocab_dict[pad_token] = len(vocab_dict)
            
        return vocab_dict

    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    data_name = DATASET.split('/')[-1] if 'mozilla' in DATASET else DATASET

    vocab_file = os.path.join('./vocab', "vocab-" + data_name + "-" + DATASET_CONFIG + ".json")

    with training_args.main_process_first(desc="dataset map vocabulary creation"):
        if not os.path.isfile(vocab_file):
            os.makedirs('vocab', exist_ok=True)
            vocab_dict = create_vocabulary_from_data(
                raw_datasets,
                word_delimiter_token=word_delimiter_token,
                unk_token=unk_token,
                pad_token=pad_token,
            )

            with open(vocab_file, "w") as file:
                json.dump(vocab_dict, file)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_file,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delimiter_token=word_delimiter_token,
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=data_args.use_auth_token
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

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
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            # "activation_dropout": model_args.activation_dropout,
            "use_weighted_layer_sum": USE_WEIGHTED_LAYER_SUM,
        }
    )

    model_cls = HubertAdapterModel if 'hubert' in MODEL_NAME else Wav2Vec2AdapterModel
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        use_auth_token=data_args.use_auth_token,
    )

    total_params = sum(p.numel() for p in model.parameters())

    if ADAPTER:
        if ADAPTER == "prefix":
            adapter_config = PrefixTuningConfig(
                flat=False,
                prefix_length=PREFIX_LENGTH,
                bottleneck_size=BOTTLENECK_SIZE,
                non_linearity="tanh",
                dropout=PREFIX_DROPOUT,
            )

        elif ADAPTER == "houlsby":
            adapter_config = HoulsbyConfig(
                ln_after=LN_AFTER,
                non_linearity=NON_LINEARITY,
                reduction_factor=REDUCTION_FACTOR,
            )
        elif ADAPTER == "pfeiffer":
            adapter_config = PfeifferConfig(
                non_linearity=NON_LINEARITY,
                reduction_factor=REDUCTION_FACTOR,
            )
        elif ADAPTER == "parallel":
            adapter_config = ParallelConfig(
                ln_after=LN_AFTER,
                non_linearity=NON_LINEARITY,
                reduction_factor=REDUCTION_FACTOR,
            )
        elif ADAPTER == "union":
            adapter_config = ConfigUnion(
                PrefixTuningConfig(
                    flat=False,
                    prefix_length=PREFIX_LENGTH,
                    bottleneck_size=BOTTLENECK_SIZE,
                    non_linearity="tanh",
                    dropout=PREFIX_DROPOUT,
                ),
                HoulsbyConfig(
                    ln_after=LN_AFTER,
                    non_linearity=NON_LINEARITY,
                    reduction_factor=REDUCTION_FACTOR,
                ),
            )
        else:
            raise ValueError("No corresponding adapter config.")
        model.add_adapter("asr-ctc", config=adapter_config, overwrite_ok=True)
        model.train_adapter("asr-ctc")
    else:
        if USE_WEIGHTED_LAYER_SUM:
            model.freeze_base_model()
        else:
            model.freeze_feature_encoder()

    model.add_ctc_head(
        head_name="asr-ctc",
        overwrite_ok=True,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.heads.parameters())

    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Head parameters: {head_params}")
    logger.info("Ratio of trainable parameters: {:.3f}%".format(trainable_params / total_params * 100))

    if ADAPTER in ["prefix", "union"]:
        import copy
        pool = model.base_model.prefix_tuning.prefix_tunings['asr-ctc']
        pool_clone = copy.deepcopy(pool)
        
        mlp_param = sum(p.numel() for p in pool_clone.parameters() if p.requires_grad)
        pool_clone.eject()
        flat_param = sum(p.numel() for p in pool_clone.parameters() if p.requires_grad)
        
        diff = mlp_param - flat_param
        del pool_clone
        
        trainable_params = trainable_params - diff
        logger.info("[Prefix] Ratio of final added parameters: {:.3f}%".format(trainable_params / total_params * 100))

    processed_data_path = os.path.join("./dataset", DATASET + "-" + DATASET_CONFIG)

    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    phoneme_language = data_args.phoneme_language
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate

    try:
        vectorized_datasets = load_from_disk(processed_data_path)
    except:
        dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate

        if dataset_sampling_rate != feature_extractor.sampling_rate:
            # datasets can take care of automatically loading and resampling the audio
            raw_datasets = raw_datasets.cast_column(
                audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
            )

        def prepare_dataset(batch):
            # load audio
            sample = batch[audio_column_name]

            inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"])
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(batch["input_values"])

            # encode targets
            additional_kwargs = {}
            if phoneme_language is not None:
                additional_kwargs["phonemizer_lang"] = phoneme_language

            with processor.as_target_processor():
                batch["labels"] = processor(batch["target_text"], **additional_kwargs).input_ids
            return batch

        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )
        
        os.makedirs(processed_data_path, exist_ok=True)
        vectorized_datasets.save_to_disk(processed_data_path)
        
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"]
    )
    vectorized_datasets = vectorized_datasets.remove_columns("input_length")

    eval_metrics = {metric: load_metric(metric) for metric in data_args.eval_metrics}

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                logger.info(str(logs))

    trainer_cls = AdapterTrainer if ADAPTER else Trainer
    trainer = trainer_cls(
        model=model,
        data_collator=data_collator,
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