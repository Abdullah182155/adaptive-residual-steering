"""Shared CPU-only fixtures (no network, no downloads)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.config import RSCConfig  # noqa: E402
from src.models.wrapper import Phi2WithRSC  # noqa: E402


class _CharBatch(dict):
    """Minimal BatchEncoding stand-in: dict + attribute access + .to(device)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        for k, v in list(self.items()):
            if torch.is_tensor(v):
                self[k] = v.to(device)
        return self


class CharTokenizer:
    """ASCII character-level tokenizer. ids 0/1/2 are pad/eos/unk; chars map to ord(c) (3..127)."""

    vocab_size = 128

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.padding_side = "right"

    def __len__(self):
        return self.vocab_size

    def _enc_char(self, c: str) -> int:
        o = ord(c)
        return o if 3 <= o < self.vocab_size else self.unk_token_id

    def encode(self, text, add_special_tokens: bool = False, **kwargs):
        ids = [self._enc_char(c) for c in text]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        out = []
        for i in ids:
            i = int(i)
            if i < 3:
                if not skip_special_tokens and i == self.unk_token_id:
                    out.append("?")
                continue
            out.append(chr(i))
        return "".join(out)

    def get_vocab(self):
        vocab = {self.pad_token: 0, self.eos_token: 1, self.unk_token: 2}
        vocab.update({chr(i): i for i in range(3, self.vocab_size)})
        return vocab

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True,
                 max_length=None, return_offsets_mapping=False, **kwargs):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        ids = [self.encode(t) for t in batch]
        offsets = [[(i, i + 1) for i in range(len(x))] for x in ids]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
            offsets = [o[:max_length] for o in offsets]

        if padding == "max_length" and max_length:
            target = max_length
        elif padding:
            target = max((len(x) for x in ids), default=0)
        else:
            target = None

        masks = [[1] * len(x) for x in ids]
        if target is not None:
            for j in range(len(ids)):
                n_pad = max(0, target - len(ids[j]))
                pad_ids, pad_mask, pad_off = [self.pad_token_id] * n_pad, [0] * n_pad, [(0, 0)] * n_pad
                if self.padding_side == "left":
                    ids[j], masks[j], offsets[j] = pad_ids + ids[j], pad_mask + masks[j], pad_off + offsets[j]
                else:
                    ids[j], masks[j], offsets[j] = ids[j] + pad_ids, masks[j] + pad_mask, offsets[j] + pad_off

        out = _CharBatch()
        if return_tensors == "pt":
            out["input_ids"] = torch.tensor(ids, dtype=torch.long)
            out["attention_mask"] = torch.tensor(masks, dtype=torch.long)
            if return_offsets_mapping:
                out["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)
        else:
            out["input_ids"] = ids[0] if single else ids
            out["attention_mask"] = masks[0] if single else masks
            if return_offsets_mapping:
                out["offset_mapping"] = offsets[0] if single else offsets
        return out


@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def tiny_cfg(tmp_path) -> RSCConfig:
    ckpt = tmp_path / "checkpoints"
    return RSCConfig(
        model_name="tiny-phi-test",
        lora_rank=4,
        lora_alpha=8.0,
        gate_dim=4,
        batch_size=1,
        grad_accum=1,
        n_samples=8,
        max_tok_len=128,
        accuracy_eval_n=4,
        phase1_epochs=1,
        router_bootstrap_epochs=1,
        router_bootstrap_n_examples=4,
        steer_scale=0.30,
        checkpoint_dir=str(ckpt),
        lora_save_path=str(ckpt / "steer_phase1.pt"),
        phase2_save_path=str(ckpt / "phase2.pt"),
        router_bootstrap_save_path=str(ckpt / "router.pt"),
        manifest_path=str(tmp_path / "run_manifest.json"),
    )


@pytest.fixture
def tiny_model(tiny_cfg) -> Phi2WithRSC:
    torch.manual_seed(0)
    base = None
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        # Check if Phi models are directly importable
        from transformers.models.phi.modeling_phi import PhiForCausalLM
        from transformers.models.phi.configuration_phi import PhiConfig
        phi_cfg = PhiConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=12,
            num_attention_heads=4,
            max_position_embeddings=256,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
        base = PhiForCausalLM(phi_cfg).float()
    except Exception:
        pass

    if base is None:
        class _DummyLayer(torch.nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
            def forward(self, hidden_states, *args, **kwargs):
                return (self.linear(hidden_states),)

        class _DummyModel(torch.nn.Module):
            def __init__(self, hidden_size=64, num_layers=12, vocab_size=128):
                super().__init__()
                self.config = type("Config", (), {
                    "num_hidden_layers": num_layers,
                    "hidden_size": hidden_size,
                    "vocab_size": vocab_size,
                })()
                self.model = type("Model", (), {
                    "layers": torch.nn.ModuleList([_DummyLayer(hidden_size) for _ in range(num_layers)])
                })()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids=None, attention_mask=None, labels=None, *args, **kwargs):
                if input_ids is not None:
                    h = self.embed(input_ids)
                else:
                    h = torch.zeros(1, 4, 64)
                for layer in self.model.layers:
                    h = layer(h)[0]
                logits = self.lm_head(h)
                loss = None
                if labels is not None:
                    loss = torch.tensor(0.5, requires_grad=True)
                class _Output:
                    pass
                out = _Output()
                out.loss = loss
                out.logits = logits
                return out

            def generate(self, input_ids=None, attention_mask=None,
                         max_new_tokens=4, do_sample=False, **kwargs):
                B = input_ids.shape[0] if input_ids is not None else 1
                extra = torch.zeros(B, max_new_tokens, dtype=torch.long)
                return torch.cat([input_ids, extra], dim=1) if input_ids is not None else extra

        base = _DummyModel(hidden_size=64, num_layers=12, vocab_size=128)

    return _finish_build(base, tiny_cfg)


def _build_tiny_model(cfg: RSCConfig) -> Phi2WithRSC:
    """Helper to build a tiny model from a given cfg (used by M6a tests)."""
    torch.manual_seed(0)
    base = None
    try:
        from transformers.models.phi.modeling_phi import PhiForCausalLM
        from transformers.models.phi.configuration_phi import PhiConfig
        phi_cfg = PhiConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=12, num_attention_heads=4,
            max_position_embeddings=256, pad_token_id=0,
            bos_token_id=1, eos_token_id=1,
        )
        base = PhiForCausalLM(phi_cfg).float()
    except Exception:
        pass
    if base is None:
        class _DL(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear = torch.nn.Linear(d, d)
            def forward(self, hidden_states, *a, **kw):
                return (self.linear(hidden_states),)
        class _DM(torch.nn.Module):
            def __init__(self, hidden_size=64, num_layers=12, vocab_size=128):
                super().__init__()
                self.config = type("Config", (), {
                    "num_hidden_layers": num_layers,
                    "hidden_size": hidden_size,
                    "vocab_size": vocab_size,
                })()
                self.model = type("Model", (), {
                    "layers": torch.nn.ModuleList([_DL(hidden_size) for _ in range(num_layers)])
                })()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                h = self.embed(input_ids) if input_ids is not None else torch.zeros(1, 4, 64)
                for layer in self.model.layers:
                    h = layer(h)[0]
                logits = self.lm_head(h)
                loss = torch.tensor(0.5, requires_grad=True) if labels is not None else None
                class _Out:
                    pass
                out = _Out(); out.loss = loss; out.logits = logits
                return out
            def generate(self, input_ids=None, attention_mask=None, max_new_tokens=4, **kw):
                # Greedy: just append zeros
                B = input_ids.shape[0] if input_ids is not None else 1
                extra = torch.zeros(B, max_new_tokens, dtype=torch.long)
                return torch.cat([input_ids, extra], dim=1) if input_ids is not None else extra
        base = _DM(hidden_size=64, num_layers=12, vocab_size=128)
    return _finish_build(base, cfg)


def _finish_build(base, cfg):
    model = Phi2WithRSC(base, cfg, device=torch.device("cpu"))
    model.eval()
    return model


@pytest.fixture
def tiny_rows(char_tokenizer) -> list:
    rows = []
    max_len = 128
    for i in range(8):
        a, b = 2 + i, 3 + (i * 2) % 7
        if i % 2 == 0:
            steps = [f"{a} + {b} = {a + b}"]
            final = str(a + b)
        else:
            steps = [f"{a} + {b} = {a + b}", f"{a + b} * 2 = {(a + b) * 2}"]
            final = str((a + b) * 2)
        question = f"What is {a} + {b}?"
        prefix = f"Question: {question}\nAnswer: Let's think step by step.\n"
        body = "\n".join(f"Step {j + 1}: {s}." for j, s in enumerate(steps))
        text = prefix + body + f"\n#### {final}"
        enc = char_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()
        prefix_len = len(char_tokenizer.encode(prefix))
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100
        rows.append(
            {
                "question": question,
                "steps_list": steps,
                "final_num": final,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return rows
