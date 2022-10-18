import numpy as np
import torch
import torch.nn.functional as F
from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.sampling_strategies.base_strategy import top_k_logits   

class BaseStrategy:
    def __init__(self, batch_size, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0, end_tokens=None):
        self.batch_size = batch_size
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems, temperature=None):
        logits = logits.view(-1, logits.size(-1))
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504

        logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(logits.float(), dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        for i in range(self.batch_size):
            if i >= batch_size:
                self._is_done[i] = True
            elif self._is_done[i]:
                pred[i] = -1
            elif pred[i].item() in self.end_tokens:
                self._is_done[i] = True
        
        tokens = torch.cat((tokens, pred.view(tokens.shape[:-1] + (1,))), dim=-1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)
        return tokens, mems


class CTGStrategy(BaseStrategy):
    def __init__(self, model,batch_size, invalid_slices=[], temperature=1., top_k=5, eps=1e-4, top_p=0.0, end_tokens=None):
        super().__init__(batch_size, invalid_slices=[], temperature=1., top_k=5, eps=1e-4, top_p=0.0, end_tokens=None)
        self.model = model
        self.alpha = 0.6 # same definition as simCTG's Contrastive Search        
        self.final_output = []
        self.beam_width = 5
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens

    def forward(self, logits, tokens, mems,index,counter,position_ids,attention_mask,context_hidden,temperature=None,**kw_args):
        logits = logits.view(-1, logits.size(-1))
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        
        logits = top_k_logits(logits, self.beam_width, self.top_p)  
        probs = F.softmax(logits.float(), dim=-1)  # float is essetial, due to a bug in Pytorch
        pred_next = torch.multinomial(probs, num_samples=self.beam_width)
        top_k_probs = torch.gather(probs, dim=1, index=pred_next)
        pred_next = pred_next.reshape(batch_size * self.beam_width,-1)
        
        position_ids = position_ids.unsqueeze(1).expand(batch_size, 1, -1).reshape(batch_size, -1)
        position_ids = torch.repeat_interleave(position_ids,self.beam_width,dim=0)
        attention_mask_shape = attention_mask.shape[-3:]
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, 1, -1, -1, -1).reshape(
                batch_size, *attention_mask_shape)
        attention_mask = torch.repeat_interleave(attention_mask,self.beam_width,dim=0)
        tmp_mems = mems.reshape(mems.shape[0], batch_size, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        tmp_mems = torch.repeat_interleave(tmp_mems,self.beam_width,dim=1) if tmp_mems is not None else None
        _,*output_per_layers = self.model(
            pred_next,
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=tmp_mems,
            **kw_args
        )
        next_hidden = output_per_layers[-1].get("hidden_states").transpose(0,1) # bsz*1*embed dim
        selected_idx = self.ranking_fast(torch.repeat_interleave(context_hidden.float(),self.beam_width,dim=0),next_hidden.float(),top_k_probs.float(),self.alpha,self.beam_width)
        selected_token = pred_next[selected_idx]  
        
        for i in range(self.batch_size):
            if i >= batch_size:
                self._is_done[i] = True
            elif self._is_done[i]:
                selected_token[i] = -1
            elif selected_token[i].item() in self.end_tokens:
                self._is_done[i] = True

        tokens = torch.cat((tokens, selected_token.unsqueeze(2)), dim=-1)
        context_hidden = torch.cat((context_hidden,next_hidden[selected_idx]),1)
        return tokens, mems, context_hidden

    def ranking_fast(self,context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
        '''
            context_hidden: bsz*beam x seqlen x embed_dim
            next_hidden: bsz*beam x 1 x embed_dim
            next_top_k_probs: bsz x beam
        '''   
        norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
        norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
        cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
        scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
        next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
        scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
        scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
        selected_idx = scores.max(dim=-1)[1]
        return selected_idx




class BeamSearchStrategy:
    def __init__(
        self,
        batch_size,
        num_beams,
        length_penalty=1.0,
        consider_end=False,
        end_tokens=[],
        invalid_slices=[],
        no_repeat_ngram_size=0,
        min_gen_length=0,
        deterministic=False,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_gen_length = min_gen_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.deterministic = deterministic
        self._init_cache()

    def _init_cache(self):
        self.end_beams = [[] for _ in range(self.batch_size)]  # list of LongTensors
        self.end_beams_penalized_scores = [[] for _ in range(self.batch_size)]  # list of LongTensors
        self.cached_beam_scores = 0  # [batch_size]
        self.cached_beam_ngram_bans = [[{} for _ in range(self.num_beams)] for _ in range(self.batch_size)]
        self.length_generated = 0
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)

    def _add_end_beams(self, score, beam, batch_idx):
        score = score / ((5.0 + len(beam)) / 6) ** self.length_penalty  # Magic number for OpenNMT
        for i in range(len(self.end_beams[batch_idx]), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[batch_idx][i - 1]:
                break
        self.end_beams[batch_idx].insert(i, beam)
        self.end_beams_penalized_scores[batch_idx].insert(i, score)

        self.end_beams[batch_idx] = self.end_beams[batch_idx][: self.num_beams]
        self.end_beams_penalized_scores[batch_idx] = self.end_beams_penalized_scores[batch_idx][: self.num_beams]

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems):
        batch_size, num_beams, vocab_size = logits.shape
        seq_len = tokens.shape[-1]
        logits = logits.float()
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if self.min_gen_length > self.length_generated:
            for end_token in self.end_tokens:
                logits[..., end_token] = -65504
        if self.ngram > 0 and seq_len > self.ngram:
            for batch_idx in range(batch_size):
                for i in range(num_beams):
                    ngram_prefix = tokens[batch_idx, i, -(self.ngram - 1) :].tolist()  # TODO ngram=1
                    for banned_index in self.cached_beam_ngram_bans[batch_idx][i].get(tuple(ngram_prefix), []):
                        logits[batch_idx, i, banned_index] = -65504

        next_token_scores = F.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        if isinstance(prev_scores, torch.Tensor):
            prev_scores = prev_scores[..., None].expand_as(next_token_scores)
        next_token_scores = next_token_scores + prev_scores

        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        probs = F.softmax(next_token_scores, dim=-1)
        if num_beams < self.num_beams:  # First token
            probs = probs[..., :vocab_size]
        if self.deterministic:
            next_tokens = torch.topk(probs, k=(max(1, len(self.end_tokens)) + 1) * self.num_beams).indices  # [2*nb]
        else:
            next_tokens = torch.multinomial(
                probs, num_samples=(max(1, len(self.end_tokens)) + 1) * self.num_beams
            )  # [2*nb]
        next_token_scores = next_token_scores[torch.arange(batch_size).unsqueeze(1), next_tokens]
        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
        next_tokens = next_tokens[torch.arange(batch_size).unsqueeze(1), _indices]

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="trunc")
        next_tokens = next_tokens % vocab_size

        # select out end beams or continue beams
        beam_continue_batch, score_continue_batch, mems_continue_batch = [], [], []
        for batch_idx in range(batch_size):
            beam_continue = []
            scores_continue = []
            bans_continue = []
            mems_contiue = []
            for i in range(len(next_tokens[batch_idx])):
                beam = torch.cat((tokens[batch_idx, next_indices[batch_idx, i]], next_tokens[batch_idx, i : i + 1]))
                if not self._is_done[batch_idx] and int(next_tokens[batch_idx, i]) in self.end_tokens:
                    self._add_end_beams(next_token_scores[batch_idx, i], beam, batch_idx)
                elif len(beam_continue) < self.num_beams:
                    beam_continue.append(beam)
                    mems_contiue.append(mems[:, batch_idx, next_indices[batch_idx, i]])
                    # update caches
                    scores_continue.append(next_token_scores[batch_idx, i])
                    if self.ngram > 0:
                        bans = self.cached_beam_ngram_bans[batch_idx][next_indices[batch_idx, i]].copy()
                        # TODO ngram=1
                        ngram_prefix = tuple(tokens[batch_idx, next_indices[batch_idx, i], -(self.ngram - 1):].tolist())
                        bans[ngram_prefix] = bans.get(ngram_prefix, tuple()) + (next_tokens[batch_idx, i],)
                        bans_continue.append(bans)
                else:
                    break
            beam_continue_batch.append(torch.stack(beam_continue))
            mems_continue_batch.append(torch.stack(mems_contiue, dim=1))
            score_continue_batch.append(scores_continue)
            self.cached_beam_ngram_bans[batch_idx] = bans_continue
        tokens = torch.stack(beam_continue_batch)
        mems = torch.stack(mems_continue_batch, dim=1)
        self.cached_beam_scores = torch.tensor(score_continue_batch, device=logits.device)
        self.length_generated += 1
        for batch_idx in range(self.batch_size):
            if batch_idx >= batch_size:
                self._is_done[batch_idx] = True
            elif (
                len(self.end_beams[batch_idx]) == self.num_beams
                and self.end_beams_penalized_scores[batch_idx][-1]
                >= self.cached_beam_scores[batch_idx].max() / ((5.0 + (seq_len + 1)) / 6) ** self.length_penalty
            ):  # We're done if none of current tokens will better than the worst in end_beams
                self._is_done[batch_idx] = True

        return tokens, mems

    def finalize(self, tokens, mems):
        if self.consider_end:
            batch_size, num_beams = tokens.shape[:2]
            for batch_idx in range(batch_size):
                if not self._is_done[batch_idx]:
                    for i in range(num_beams):
                        self._add_end_beams(self.cached_beam_scores[batch_idx, i], tokens[batch_idx, i], batch_idx)
            mems = None
            ret = self.end_beams[:batch_size]
        else:
            ret = tokens
        self._init_cache()
        return ret, mems
