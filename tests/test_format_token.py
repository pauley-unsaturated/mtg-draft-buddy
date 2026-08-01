"""Format-token additions to DeckBuilder (P5.S joint program).

The contract: formats=False models are byte-identical to before (old
checkpoints unaffected); formats=True models are format-blind at init
(zero-init embedding) so a warm start reproduces the source exactly, and
only diverge per-format once the embedding trains.
"""

import torch

from draftbot.models.builder import DeckBuilder


def _model(**kw):
    torch.manual_seed(3)
    feats = torch.randn(19, 11)
    return DeckBuilder(feats, emb_dim=32, heads=4, blocks=1, dropout=0.0, **kw)


def _pool():
    ids = torch.tensor([[1, 4, 7, 12, -1]])
    cnt = torch.tensor([[1, 2, 1, 1, 0]])
    return ids, cnt, ids == -1


def test_formats_off_has_no_format_params_and_ignores_format_id():
    m = _model()
    assert not any("format_emb" in k for k in m.state_dict())
    ids, cnt, pad = _pool()
    m.eval()
    a = m(ids, cnt, pad, format_id=0)
    b = m(ids, cnt, pad, format_id=1)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_formats_zero_init_is_format_blind():
    m = _model(formats=True).eval()
    ids, cnt, pad = _pool()
    a = m(ids, cnt, pad, format_id=0)
    b = m(ids, cnt, pad, format_id=1)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_trained_format_emb_diverges_only_its_format():
    m = _model(formats=True).eval()
    ids, cnt, pad = _pool()
    base = m(ids, cnt, pad, format_id=0)
    with torch.no_grad():
        m.format_emb.weight[1] += 0.5
    still = m(ids, cnt, pad, format_id=0)
    moved = m(ids, cnt, pad, format_id=1)
    for x, y in zip(base, still):
        assert torch.equal(x, y)  # draft path untouched
    assert not torch.equal(base[0], moved[0])  # sealed path shifted


def test_warm_start_from_formatless_checkpoint():
    src = _model()
    dst = _model(formats=True)
    state = src.state_dict()
    state.pop("card_features")
    missing, unexpected = dst.load_state_dict(state, strict=False)
    assert not unexpected
    assert all("format_emb" in k or k == "card_features" for k in missing)
    ids, cnt, pad = _pool()
    src.eval(), dst.eval()
    a, b = src(ids, cnt, pad), dst(ids, cnt, pad, format_id=1)
    for x, y in zip(a, b):
        assert torch.equal(x, y)  # zero-init: warm start is a no-op
