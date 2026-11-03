"""Frozen-source state KL, with valid-action masking and graph-equal reduction."""
import torch
from .budget_feedback_torch import distributions

def source_kl(current, reference, decisions):
    ref,mask=distributions(reference.detach(),decisions)
    cur,currentmask=distributions(current,decisions)
    assert torch.equal(mask,currentmask)
    # Replace padded -inf values BEFORE arithmetic to prevent 0*(-inf- -inf).
    safe_ref=ref.masked_fill(~mask,0.)
    safe_cur=cur.masked_fill(~mask,0.)
    action_kl=(ref.exp()*(safe_ref-safe_cur)).sum(-1)
    active=mask.sum(-1)>1
    return action_kl[active].mean() if bool(active.any()) else current.sum()*0.
