import math
from dataclasses import dataclass, field
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class FocalLossCriterionConfig(FairseqDataclass):
    gamma: float = field(
        default=2.0,
        metadata={"help": "Focusing parameter gamma for focal loss"},
    )
    alpha: str = field(
        default="", metadata={"help": "Comma-separated list of class weights (alpha)"}
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "Report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

def focal_loss(lprobs, target, gamma, alpha, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    
    nll_loss = -lprobs.gather(dim=-1, index=target)
    pt = torch.exp(-nll_loss)  # Probabilities of the correct class
    focal_factor = (1 - pt) ** gamma
    
    if alpha is not None:
        alpha_factor = alpha[target.squeeze()].to(lprobs.device)
        nll_loss = alpha_factor * focal_factor * nll_loss
    else:
        nll_loss = focal_factor * nll_loss
    
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    
    if reduce:
        nll_loss = nll_loss.sum()
    
    return nll_loss

@register_criterion("focal_loss", dataclass=FocalLossCriterionConfig)
class FocalLossCriterion(FairseqCriterion):
    def __init__(
        self, 
        task, 
        sentence_avg, 
        gamma, 
        alpha, 
        ignore_prefix_size=0, 
        report_accuracy=False
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.gamma = gamma
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        
        if alpha:
            self.alpha = torch.tensor([float(a) for a in alpha.split(",")])
        else:
            self.alpha = None

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        return loss, sample_size, logging_output
    
    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
    
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        return focal_loss(lprobs, target, self.gamma, self.alpha, ignore_index=self.padding_idx, reduce=reduce)
    
    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(sum(log.get("n_correct", 0) for log in logging_outputs))
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                ) if meters["total"].sum > 0 else float("nan"),
            )
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
