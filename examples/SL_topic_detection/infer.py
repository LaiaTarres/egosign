import ast
import os
import sys
from dataclasses import dataclass, field, is_dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from operator import attrgetter
from collections import Counter
import random

import torch
import torch.distributed as dist

import hydra
from hydra.core.config_store import ConfigStore
from fairseq import checkpoint_utils, distributed_utils, progress_bar, tasks, utils
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.logging.progress_bar import BaseProgressBar
from fairseq.models.fairseq_model import FairseqModel
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "config_pose"
# TODO: This we have hardcoded here

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


class InferenceProcessor:
    cfg: InferConfig

    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.cfg.task.data = os.environ.get('DATA', cfg.task.data)
        self.cfg.task.dict_path = os.environ.get('DICT_PATH', cfg.task.dict_path)
        self.cfg.task.feats_type = os.environ.get('FEATS_TYPE', cfg.task.feats_type)
        self.cfg.task.dataset = os.environ.get('DATASET', cfg.task.dataset)
        self.cfg.common_eval.path = os.environ.get('MODEL_PATH', cfg.common_eval.path)
        self.cfg.bpe.sentencepiece_model = os.environ.get('SP_MODEL', cfg.bpe.sentencepiece_model)
        self.cfg.hooks.out_file = os.environ.get('OUTPUTS_FILE', self.cfg.hooks.out_file)
        
        self.task = tasks.setup_task(cfg.task)

        self.att_time = []
        self.embeddings = []
        self.logits = []
        model, saved_cfg = self.load_model() #aquí el problema és que ho fem amb això buit?
        self.targets = []
        self.preds = []
        self.per_video_id_prediction = {}

        self.model = model
        if torch.cuda.is_available():
            model.cuda()
        self.saved_cfg = saved_cfg
        self.src_dict = self.task.source_dictionary  # this is None except when cfg.feats_type == 'text'
        self.tgt_dict = self.task.target_dictionary  # this is always None

        self.cfg.dataset.dataset_split = os.environ.get('DATASET_SPLIT', self.cfg.dataset.dataset_split)
        
        #Here maybe we should load from .tsv file...
        self.task.load_dataset(
            self.cfg.dataset.dataset_split, 
            task_cfg=saved_cfg.task,
        )
        self.inference_timer = StopwatchMeter()
        self.wps_meter = TimeMeter()
        self.frames = 0
        self.total_counts = 0
        self.total_totals = 0

        self.progress_bar = self.build_progress_bar()

    def __enter__(self) -> "InferenceProcessor":

        logger.info(
            "num. shared model params: {:,} (num. trained: {:,})".format(
                sum(
                    p.numel() for p in self.model.parameters() if not getattr(p, "expert", False)
                ),
                sum(
                    p.numel()
                    for p in self.model.parameters()
                    if not getattr(p, "expert", False) and p.requires_grad
                ),
            )
        )

        logger.info(
            "num. expert model params: {} (num. trained: {})".format(
                sum(p.numel() for p in self.model.parameters() if getattr(p, "expert", False)),
                sum(
                    p.numel()
                    for p in self.model.parameters()
                    if getattr(p, "expert", False) and p.requires_grad
                ),
            )
        )
        return self

    def __exit__(self, *exc) -> bool:
        # TODO: right before exiting, the attention maps and embeddings in self.att_time and self.embeddings should be stored in disk

        # store targets, embeddings and attention weights for later usage (e.g. visualization)
        
        outputs = {}
        self.targets = torch.cat(self.targets, 0).squeeze()
        outputs['targets'] = self.targets
        self.preds = torch.cat(self.preds, 0).squeeze()
        outputs['preds'] = self.preds

        if self.cfg.hooks.embedding:
            self.embeddings = torch.cat(self.embeddings, 0).squeeze()
            outputs['embeddings'] = self.embeddings

        if self.cfg.hooks.attention:
            outputs['att_time'] = self.att_time

        if self.cfg.hooks.logits:
            self.logits = torch.cat(self.logits, 0).squeeze()
            outputs['logits'] = self.logits


        tsv_path = self.cfg.hooks.out_file.replace('.pt', '_vid2pred.tsv')
        print(f"saving video_id to pred in a .tsv file: {tsv_path}")
        with open(tsv_path, 'w') as f:
            f.write("video_id\tprediction\n")
            for vid, pred in self.per_video_id_prediction.items():
                f.write(f"{vid}\t{pred}\n")
        torch.save(outputs, self.cfg.hooks.out_file)
        return False

    def __iter__(self) -> Any:
        for sample in self.progress_bar:
            if not self.cfg.common.cpu or torch.cuda.is_available():
                sample = utils.move_to_cuda(sample)

            # Happens on the last batch.
            if "net_input" not in sample:
                continue
            yield sample

    def log(self, *args, **kwargs):
        self.progress_bar.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.progress_bar.print(*args, **kwargs)

    def register_hooks(self, model: FairseqModel):

        def get_input(container: List) -> Callable:
            def hook(module, input):
                container.append(input[0].detach())
            return hook

        def get_logits(container: List) -> Callable:
            softm = torch.nn.Softmax(dim=1)
            def hook(module, input, output):
                logits = output.squeeze() if len(output.shape) > 2 else output
                logits = logits.unsqueeze(0) if len(logits.shape) == 1 else logits
                logits = softm(logits)
                container.append(logits.detach())
            return hook

        # used for retrieving att over final embeddings in LSTM and transformer
        def get_output_att1(container: List) -> Callable:
            def hook(module, input, output):
                container.append(output[1].detach())
            return hook

        # used for retrieving encoder cross-attention in PerceiverIO
        def get_output_crossatt_PerceiverIO(container: List) -> Callable:
            def hook(module, input, output):
                att = output[3][0].detach()
                if len(att.shape) != 4:
                    raise RuntimeError(
                        (f'Expected input embeddings to be four-dimensional tensor'
                        f' but got a `{len(att.shape)}-dimensional tensor instead.`')
                    )
                att = torch.mean(att, (1, 2))
                container.append(att)
            return hook

        if self.cfg.hooks.attention:  # register a hook to retrieve attention maps over the input sequence
            # TODO: set these layers
            layers_attention_time = {
                'PerceiverModel': 'encoder',
                'Sign2TextTransformerModel': 'att_encoder',
                'SLTopicDetectionLSTMModel': 'att_encoder',
            }
            hooks_time = {
                'PerceiverModel': get_output_crossatt_PerceiverIO,
                'Sign2TextTransformerModel': get_output_att1,
                'SLTopicDetectionLSTMModel': get_output_att1,
            }
            # TODO: decide how to store these results. Should they be stored in a variable and then to disk after passing through the whole dataset?
            #       or should they rather be stored one-by-one/batch-by-batch?
            model_class = model.__class__.__name__
            if model_class == 'Sign2TextTransformerModel_CLS':
                raise AttributeError('Cannot visualize attention with this script for model class `Sign2TextTransformerModel_CLS`.')
            retriever = attrgetter(layers_attention_time[model_class])
            retriever(model).register_forward_hook(hooks_time[model_class](self.att_time))

        if self.cfg.hooks.embedding:  # register hook to retrieve embeddings produced at the last layer before the classification head
            layers_embedding = {
                'PerceiverModel': 'decoder.decoder.decoding_cross_attention.attention.output',
                'Sign2TextTransformerModel': 'classif_head',
                'Sign2TextTransformerModel_CLS': 'classif_head',  # TODO: hook for retrieving CLS token
                'SLTopicDetectionLSTMModel': 'classif_head',
            }
            # for name, layer in model.named_modules():
            #     print(name, layer)
            # print(f'model.__class__.__name__ {model.__class__.__name__}')
            retriever = attrgetter(layers_embedding[model.__class__.__name__])
            retriever(model).register_forward_pre_hook(get_input(self.embeddings))

        if self.cfg.hooks.logits:
            layers_logits = {
                'PerceiverModel': 'decoder.decoder.final_layer',
                'Sign2TextTransformerModel': 'classif_head',
                'Sign2TextTransformerModel_CLS': 'classif_head',
                'SLTopicDetectionLSTMModel': 'classif_head',
            }
            retriever = attrgetter(layers_logits[model.__class__.__name__])
            retriever(model).register_forward_hook(get_logits(self.logits))

    def load_model(self) -> Tuple[FairseqModel, FairseqDataclass]:
        arg_overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        logger.info("Loading model from %s", self.cfg.common_eval.path)
        models, saved_cfg = checkpoint_utils.load_model_ensemble( #The error might be here which parameters are loaded. I don't understand how it changes...
            utils.split_paths(self.cfg.common_eval.path, separator="\\"),
            arg_overrides=arg_overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )
        logger.info(models[0])
        self.register_hooks(models[0])
        return models[0], saved_cfg

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        #Here I need to check this code, where does it come from?? And why does it have this KeyError
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.dataset_split),
            max_tokens=self.cfg.dataset.max_tokens,
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        ).next_epoch_itr(shuffle=False)

    def build_progress_bar(
        self,
        epoch: Optional[int] = None,
        prefix: Optional[str] = None,
        default_log_format: str = "tqdm",
    ) -> BaseProgressBar:
        return progress_bar.progress_bar(
            iterator=self.get_dataset_itr(),
            log_format=self.cfg.common.log_format,
            log_interval=self.cfg.common.log_interval,
            epoch=epoch,
            prefix=prefix,
            tensorboard_logdir=self.cfg.common.tensorboard_logdir,
            default_log_format=default_log_format,
        )

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.inference_timer.start()
        counts, total = self.task.inference_step(
            sample,
            self.model,
            output_attentions=(self.model.__class__.__name__=='PerceiverModel' and self.cfg.hooks.attention),
            targets_container=self.targets,
            preds_container=self.preds,
        )
        self.inference_timer.stop(total)
        self.total_counts += counts
        self.total_totals += total
        self.frames = max(sample['net_input']['src_tokens'].shape[1], self.frames)
        
        #In here we have: self.preds with the predictions, and self.targets with the targets. We want to compute one confusion matrix per signer, and store it on a variable.
        # The signer_id can be found with sample['vid_id'].split('_')[-1] -> this will likely be a list
        #We want to accumulate this. 
        
        # Initialize confusion matrices dict if it doesn't exist
        if not hasattr(self, 'signer_confusion_matrices'):
            self.signer_confusion_matrices = {}
        # Compute per-signer confusion matrices and accumulate them
        num_classes = 10
        for i, vid_id in enumerate(sample['vid_id']):
            signer_id = vid_id.split('-')[-1]
            pred = self.preds[-1][i].item()
            target = self.targets[-1][i].item()
            
            self.per_video_id_prediction[vid_id] = pred 

            if signer_id not in self.signer_confusion_matrices:
                self.signer_confusion_matrices[signer_id] = torch.zeros((num_classes, num_classes), dtype=torch.long)

            self.signer_confusion_matrices[signer_id][target, pred] += 1
        
        
    def log_generation_time(self) -> None:
        logger.info(
            "Processed %d samples (the longest one having %d frames) in %.1fs %.2f,"
            " frames per second, %.2f samples per second)",
            self.inference_timer.n,
            self.frames,
            self.inference_timer.sum,
            self.frames * self.inference_timer.n / self.inference_timer.sum,
            1.0 / self.inference_timer.avg * self.frames,
        )


def parse_acc(wer_file: Path) -> float:
    with open(wer_file, "r") as f:
        return float(f.readline().strip().split(" ")[1])


def get_acc_file(cfg: InferConfig) -> Path:
    """Hashes the decoding parameters to a unique file ID."""
    base_path = "acc"
    return Path(base_path)

import numpy as np
def compute_metrics_from_confusion(conf_matrix):
    """
    Computes accuracy, precision, recall, and F1-score (macro-averaged) from a confusion matrix.
    Assumes conf_matrix is a square torch tensor (num_classes x num_classes).
    """
    conf_matrix = conf_matrix.numpy()
    num_classes = conf_matrix.shape[0]

    # True positives, predicted positives, and actual positives per class
    TP = np.diag(conf_matrix)
    predicted_positives = np.sum(conf_matrix, axis=0)  # sum over columns
    actual_positives = np.sum(conf_matrix, axis=1)     # sum over rows

    # Per-class metrics (handle division by zero)
    precision_per_class = np.divide(TP, predicted_positives, out=np.zeros_like(TP, dtype=float), where=predicted_positives != 0)
    recall_per_class = np.divide(TP, actual_positives, out=np.zeros_like(TP, dtype=float), where=actual_positives != 0)
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(TP, dtype=float),
        where=(precision_per_class + recall_per_class) != 0
    )

    # Macro averages
    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = np.mean(f1_per_class)

    # Accuracy
    total_correct = np.sum(TP)
    total = np.sum(conf_matrix)
    accuracy = total_correct / total if total > 0 else 0.0

    return accuracy, precision, recall, f1


def main(cfg: InferConfig) -> float:
    """Entry point for main processing logic.

    Args:
        cfg: The inference configuration to use.
        acc: Optional shared memory pointer for returning the accuracy. If not None,
            the final accuracy value will be written here instead of being returned.

    Returns:
        The final accuracy if `acc` is None, otherwise None.
    """
    acc_file = get_acc_file(cfg)

    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 20000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")

    with InferenceProcessor(cfg) as processor:
        i = 1
        for sample in processor:
            processor.process_sample(sample)
            i += 1

        processor.log_generation_time()
        
        for signer_id in sorted(processor.signer_confusion_matrices.keys()):
            matrix = processor.signer_confusion_matrices[signer_id]
            logger.info(f"Signer ID: {signer_id}\n{matrix}")
        
        signer_ids = sorted(processor.signer_confusion_matrices.keys())
        metrics_per_signer = []
        for signer_id in signer_ids:
            matrix = processor.signer_confusion_matrices[signer_id]
            accuracy, precision, recall, f1 = compute_metrics_from_confusion(matrix)
            metrics_per_signer.append({
                'signer_id': signer_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            logger.info(f"Signer ID: {signer_id}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        counts_t, totals_t = processor.total_counts, processor.total_totals

        if cfg.common.cpu:
            logger.warning("Merging Accuracy requires CUDA.")
            if type(counts_t) != int or type(counts_t) != int:
                raise RuntimeError(
                    (f'counts of samples of type `{type(counts_t), type(counts_t)}`'
                     ' are not of type `int`')
                )
        elif processor.data_parallel_world_size > 1:
            stats = torch.LongTensor([counts_t, totals_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            counts_t, totals_t = stats[0].item(), stats[1].item()

        acc = counts_t * 100.0 / totals_t
        
        if distributed_utils.is_master(cfg.distributed_training):
            with open(acc_file, "w") as f:
                f.write(
                    (
                        f"Accuracy: {acc}\n"
                        f"counts / total = {counts_t} / {totals_t}\n\n"
                    )
                )

        return acc


@hydra.main(config_path=config_path, config_name=os.environ['CONFIG_NAME'])  # TODO: set this systematically
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    acc = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

        acc = parse_acc(get_acc_file(cfg))
    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    logger.info("Accuracy: %.4f", acc)
    if cfg.is_ax:
        return acc, None

    return acc

def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
        print(f'cfg_name = {cfg_name}')
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"
    
    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)
            
    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
