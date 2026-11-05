"""keys / warmup / every / prior semantics shared by all balancers."""
import pytest, torch
from balloonlib.MOO_Loss import SoftAdapt, ReLoBRaLo, InverseGradNorm, LRAnnealing


def _setup():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1); x = torch.randn(20, 3)
    y = model(x)
    ld = {"ode": (y ** 2).mean(), "bold": ((y - 1) ** 2).mean(), "ic": y.abs().mean(), "other": (y ** 4).mean()}
    ld["total"] = sum(ld.values())
    lw = {k: [1.0] for k in ("ode", "bold", "ic", "other")}
    return model, ld, lw


@pytest.mark.parametrize("make", [lambda: SoftAdapt(keys=["ode", "bold"], n=2), lambda: ReLoBRaLo(keys=["ode", "bold"], seed=0),
                                  lambda: InverseGradNorm(keys=["ode", "bold"], every=1),
                                  lambda: LRAnnealing(ref="bold", keys=["ode", "bold"], stat="mean", every=1)])
def test_subset_keys_leave_other_terms_fixed(make):
    model, ld, lw = _setup(); bal = make()
    for i in range(3):
        bal(ld, lw, i, model=model)
    assert len(lw["ic"]) == 1 and len(lw["other"]) == 1           # untouched
    assert len(lw["ode"]) > 1 and len(lw["bold"]) > 1             # balanced
    assert set(bal.history[-1]["weights"]) == {"ode", "bold"}


def test_default_keys_are_all_but_total():
    model, ld, lw = _setup(); bal = InverseGradNorm(every=1)
    bal(ld, lw, 0, model=model)
    assert set(bal.history[-1]["weights"]) == {"ode", "bold", "ic", "other"}


def test_total_and_missing_keys_rejected():
    model, ld, lw = _setup()
    with pytest.raises(ValueError):
        InverseGradNorm(keys=["total"], every=1)(ld, lw, 0, model=model)
    with pytest.raises(KeyError):
        InverseGradNorm(keys=["nope"], every=1)(ld, lw, 0, model=model)


def test_warmup_every_and_prior():
    model, ld, lw = _setup(); bal = InverseGradNorm(every=3, warmup=2, prior={"ode": 2.0})
    done = [i for i in range(10) if bal(ld, lw, i, model=model) is not None]
    assert done == [2, 5, 8]
    plain = InverseGradNorm(every=1)(ld, None, 0, model=model)
    assert bal.history[0]["weights"]["ode"] == pytest.approx(2.0 * plain["ode"])
    assert bal.history[0]["weights"]["bold"] == pytest.approx(plain["bold"])


def test_docstrings_mention_keys():
    for cls in (SoftAdapt, ReLoBRaLo, InverseGradNorm, LRAnnealing):
        assert "keys : list of str or None" in cls.__doc__


@pytest.mark.parametrize("make", [lambda P: InverseGradNorm(every=1, params=P),
                                  lambda P: LRAnnealing(ref="bold", stat="mean", every=1, params=P)])
def test_params_generator_survives_several_updates(make):
    model, ld, lw = _setup()
    bal = make(model.parameters())                      # generator
    w = [bal(ld, None, i, model=model) for i in range(3)]
    assert all(x is not None for x in w)
    ref = make(list(model.parameters()))
    assert w[-1] == pytest.approx([ref(ld, None, i, model=model) for i in range(3)][-1])
