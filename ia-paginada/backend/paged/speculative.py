"""Decodificação especulativa sem perda de qualidade.

Num modelo paginado, cada passada lê TODAS as páginas do disco; conferir 6
tokens numa passada custa quase o mesmo que gerar 1. Então:

1. Propomos tokens prováveis sem outro modelo ("prompt lookup"): procuramos o
   final do texto atual mais atrás no próprio texto e copiamos o que veio depois.
2. O modelo grande confere todos de uma vez (uma passada).
3. Aceitamos os palpites enquanto batem com o que o modelo escolheria; o
   primeiro que não bate é trocado pela escolha do modelo.

O resultado é IDÊNTICO ao da geração normal (greedy: mesmos tokens; com
amostragem: cada token continua sorteado da distribuição do modelo). Só fica
mais rápido quando os palpites acertam (código, resumos, reescritas, listas...).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch


def propose(ids: Sequence[int], max_draft: int, ngram_max: int = 3, ngram_min: int = 1) -> List[int]:
    """Palpites copiando o que veio depois da última ocorrência do n-grama final."""
    n_tot = len(ids)
    for n in range(min(ngram_max, n_tot - 1), ngram_min - 1, -1):
        tail = list(ids[-n:])
        for start in range(n_tot - n - 1, -1, -1):  # mais recente primeiro
            if list(ids[start:start + n]) == tail:
                follow = list(ids[start + n:start + n + max_draft])
                if follow:
                    return follow
    return []


# ----------------------------------------------------------------- caches
class _Rollback:
    """Sabe desfazer os últimos tokens de um cache depois de uma conferência."""

    def __init__(self, cache):
        self.cache = cache

    @staticmethod
    def supports(cache) -> bool:
        if cache is None:
            return False
        if hasattr(cache, "recurrent_states") and hasattr(cache, "conv_states"):
            # cache do Kimi: só dá para desfazer com a KDA em CPU (que grava as entradas)
            import sys
            return getattr(sys.modules.get("fla"), "__version__", "") == "cpu-shim"
        return hasattr(cache, "crop")

    def begin(self):
        """Chamado antes da passada de conferência."""
        c = self.cache
        if hasattr(c, "recurrent_states"):
            from . import fla_cpu
            self.snap_rec = list(c.recurrent_states)
            self.snap_conv = list(c.conv_states)
            self.snap_len = c.get_seq_length()
            fla_cpu.RECORDER = fla_cpu.Recorder()
        else:
            self.snap_len = c.get_seq_length()

    def end(self, keep: int, fed: int):
        """Mantém só ``keep`` dos ``fed`` tokens da conferência."""
        c = self.cache
        if hasattr(c, "recurrent_states"):
            from . import fla_cpu
            rec, fla_cpu.RECORDER = fla_cpu.RECORDER, None
            if keep == fed:
                return
            L = self.snap_len + keep
            for i in range(len(c.key_cache)):
                if c.key_cache[i] is not None:
                    c.key_cache[i] = c.key_cache[i][:, :, :L]
                    c.value_cache[i] = c.value_cache[i][:, :, :L]
            kda_layers = [i for i, t in enumerate(c.layer_types) if t == "linear_attention"]
            for j, layer in enumerate(kda_layers):
                c.recurrent_states[layer] = fla_cpu.replay_kda_state(
                    rec.kda[j], keep, self.snap_rec[layer])
                prev = self.snap_conv[layer] or (None, None, None)
                xs = rec.conv[3 * j:3 * j + 3]
                width = next((p.shape[-1] for p in prev if p is not None), 4)
                c.conv_states[layer] = tuple(
                    fla_cpu.replay_conv_state(x, keep, p, width) for x, p in zip(xs, prev))
        else:
            if keep != fed:
                c.crop(-(fed - keep))  # negativo = remove N tokens (transformers 4.x e 5.x)

    def abort(self):
        from . import fla_cpu
        fla_cpu.RECORDER = None


# --------------------------------------------------------------- geração
def _pick(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    logits = logits.float()
    if not temperature or temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits / float(temperature), -1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < float(top_p)
        probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
        probs = probs / probs.sum(-1, keepdim=True)
    return int(torch.multinomial(probs, 1))


def speculative_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                         temperature: float = 0.0, top_p: float = 1.0, eos_ids=(),
                         stop: Optional[Callable] = None, on_token: Optional[Callable] = None,
                         max_draft: int = 6, stats: Optional[dict] = None,
                         proposer: Optional[Callable] = None) -> torch.Tensor:
    """Geração com palpites conferidos em lote (resultado idêntico à normal)."""
    eos = set(int(e) for e in eos_ids if e is not None)
    ids: List[int] = input_ids[0].tolist()
    n0 = len(ids)
    st = stats if stats is not None else {}
    st.update(passes=0, drafted=0, accepted=0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        st["passes"] += 1
        past = out.past_key_values
        nxt = _pick(out.logits[0, -1], temperature, top_p)
        ids.append(nxt)
        if on_token:
            on_token(torch.tensor([ids]))
        can_rollback = _Rollback.supports(past)
        k = max(1, max_draft // 2)  # tamanho adaptativo do palpite
        while len(ids) - n0 < max_new_tokens and nxt not in eos:
            if stop is not None and stop(torch.tensor([ids])):
                break
            room = max_new_tokens - (len(ids) - n0)
            prop = proposer or propose
            draft = prop(ids, min(k, room - 1)) if can_rollback and room > 1 else []
            feed = [ids[-1]] + draft
            rb = _Rollback(past) if draft else None
            if rb:
                rb.begin()
            try:
                out = model(input_ids=torch.tensor([feed]), past_key_values=past, use_cache=True)
            except BaseException:
                if rb:
                    rb.abort()
                raise
            st["passes"] += 1
            past = out.past_key_values
            logits = out.logits[0]
            accepted = 0
            new: List[int] = []
            for i in range(len(feed)):
                tok = _pick(logits[i], temperature, top_p)
                new.append(tok)
                if i < len(draft) and tok == draft[i] and tok not in eos:
                    accepted += 1
                    continue
                break
            if rb:
                rb.end(keep=1 + accepted, fed=len(feed))
            st["drafted"] += len(draft)
            st["accepted"] += accepted
            if draft:  # acertou tudo: arrisca mais; errou cedo: arrisca menos
                k = min(max_draft, k + 1) if accepted == len(draft) else max(1, accepted or k // 2)
            for tok in new[:room]:
                ids.append(tok)
                if tok in eos:
                    break
            nxt = ids[-1]
            if on_token:
                on_token(torch.tensor([ids]))
    return torch.tensor([ids])
