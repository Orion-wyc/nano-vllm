import torch
from torch import nn

class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, seqs):
        results = []
        for scores, seq in zip(logits.float(), seqs):
            scores = scores.clone()
            if seq.repetition_penalty != 1.0 and seq.token_ids:
                ids = torch.tensor(list(set(seq.token_ids)), dtype=torch.long, device=scores.device)
                selected = scores.index_select(0, ids)
                selected = torch.where(selected < 0, selected * seq.repetition_penalty, selected / seq.repetition_penalty)
                scores.index_copy_(0, ids, selected)
            if seq.temperature == 0:
                results.append(scores.argmax())
                continue
            scores = scores / seq.temperature
            if seq.top_k > 0 and seq.top_k < scores.numel():
                threshold = torch.topk(scores, seq.top_k).values[-1]
                scores = scores.masked_fill(scores < threshold, float("-inf"))
            if seq.top_p < 1.0:
                sorted_scores, indices = torch.sort(scores, descending=True)
                sorted_probs = torch.softmax(sorted_scores, dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs >= seq.top_p
                sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                scores = torch.full_like(scores, float("-inf")).scatter(0, indices, sorted_scores)
            probs = torch.softmax(scores, dim=-1)
            results.append(torch.multinomial(probs, 1).squeeze(0))
        return torch.stack(results)
