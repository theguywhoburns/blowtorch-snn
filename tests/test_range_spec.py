import warnings

import pytest
import torch

from blowtorch_snn import LIF, StateSpec, RangeSpec


class _RangedLIF(LIF):
    def __init__(
        self,
        *,
        mem_range: RangeSpec,
        validate=None,
        init_hidden: bool = False,
    ):
        super().__init__(
            beta=0.9, init_hidden=init_hidden, validate=validate
        )
        self._mem_range = mem_range

    def _get_state_specs(self) -> tuple[StateSpec, ...]:
        return (
            StateSpec("spk", 0.0, differentiable=False),
            StateSpec("mem", 0.0, range=self._mem_range),
        )


def test_range_clamps_without_validation():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=True),
        validate=False,
    )

    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    _, next_mem = neuron(x, mem)

    assert torch.all(next_mem <= 1.0)


def test_range_raises_with_validation_when_error_enabled():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=True, error=True),
        validate=True,
    )

    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="violated range"):
        neuron(x, mem)


def test_range_warns_with_validation_when_warn_enabled():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=True, warn=True),
        validate=True,
    )

    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    with pytest.warns(RuntimeWarning, match="violated range"):
        _, next_mem = neuron(x, mem)

    assert torch.all(next_mem <= 1.0)


def test_range_diagnostics_disabled_when_validate_off():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=True, error=True),
        validate=False,
    )

    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, next_mem = neuron(x, mem)
    assert torch.all(next_mem <= 1.0)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_range_soft_clamp_does_not_clamp():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=False),
        validate=False,
    )

    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    _, next_mem = neuron(x, mem)

    assert torch.all(next_mem > 1.0)


def test_legacy_tuple_range_means_clamp_only():
    class LegacyLIF(LIF):
        def _get_state_specs(self) -> tuple[StateSpec, ...]:
            return (
                StateSpec("spk", 0.0, differentiable=False),
                StateSpec("mem", 0.0, value_range=(-1.0, 1.0)),
            )

    neuron = LegacyLIF(beta=0.9, validate=True)
    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, next_mem = neuron(x, mem)

    assert torch.all(next_mem <= 1.0)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_legacy_soft_range_means_no_enforcement():
    class SoftLIF(LIF):
        def _get_state_specs(self) -> tuple[StateSpec, ...]:
            return (
                StateSpec("spk", 0.0, differentiable=False),
                StateSpec(
                    "mem", 0.0, value_range=(-1.0, 1.0), soft_range=True
                ),
            )

    neuron = SoftLIF(beta=0.9, validate=True)
    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    _, next_mem = neuron(x, mem)

    assert torch.all(next_mem > 1.0)


def test_range_spec_requires_a_bound():
    with pytest.raises(ValueError, match="at least one bound"):
        RangeSpec()


def test_range_spec_rejects_low_above_high():
    with pytest.raises(ValueError, match="low must be <= high"):
        RangeSpec(low=2.0, high=-2.0)


def test_reset_value_outside_error_range_raises():
    class BadResetLIF(LIF):
        def _get_state_specs(self) -> tuple[StateSpec, ...]:
            return (
                StateSpec("spk", 0.0, differentiable=False),
                StateSpec(
                    "mem",
                    5.0,
                    range=RangeSpec(low=-1.0, high=1.0, error=True),
                ),
            )

    with pytest.raises(ValueError, match="reset_value"):
        BadResetLIF(beta=0.9)(torch.zeros(2, 4))


def test_reset_value_outside_warn_range_warns():
    class WarnResetLIF(LIF):
        def _get_state_specs(self) -> tuple[StateSpec, ...]:
            return (
                StateSpec("spk", 0.0, differentiable=False),
                StateSpec(
                    "mem",
                    5.0,
                    range=RangeSpec(low=-1.0, high=1.0, warn=True),
                ),
            )

    with pytest.warns(RuntimeWarning, match="reset_value"):
        WarnResetLIF(beta=0.9)(torch.zeros(2, 4), torch.zeros(2, 4))


def test_range_enforcer_selected_and_applies():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=True),
        validate=False,
    )
    # Metadata is built on first forward; force it so the enforcer is visible.
    neuron(torch.zeros(2, 4), torch.zeros(2, 4))
    assert neuron._range_enforcer is not None

    out = (torch.zeros(2, 4), torch.full((2, 4), 7.0))
    _, clamped = neuron._range_enforcer(out)
    assert torch.all(clamped <= 1.0)


def test_compiled_range_path_stays_clean():
    neuron = _RangedLIF(
        mem_range=RangeSpec(low=-1.0, high=1.0, clamp=True),
        validate=False,
    )
    compiled = torch.compile(neuron)

    x = torch.full((2, 4), 10.0)
    mem = torch.zeros(2, 4)

    _, next_mem = compiled(x, mem)
    assert torch.all(next_mem <= 1.0)
