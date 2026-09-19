import torch
from phenoniche.utils.validation import survival_vectors


def cox_breslow_loss(risk_score, time, event):
    survival_vectors(risk_score, time, event)
    order = torch.argsort(time, descending=True)
    ordered_time = time[order]
    ordered_risk = risk_score[order]
    ordered_event = event[order].to(risk_score.dtype)
    log_cumulative_risk = torch.logcumsumexp(ordered_risk, dim=0)
    _, counts = torch.unique_consecutive(ordered_time, return_counts=True)
    group_ends = torch.cumsum(counts, dim=0) - 1
    denominators = torch.repeat_interleave(log_cumulative_risk[group_ends], counts)
    events = ordered_event.sum().clamp_min(1)
    return -((ordered_risk - denominators) * ordered_event).sum() / events
