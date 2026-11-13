"""Tests for the options of the local facts."""

from ebaml.local import fact_types as fact_types_module


def test_local_options_defaults() -> None:
    options = fact_types_module.LocalOptions()
    assert (options.confidence_level, options.max_steps, options.n_samples, options.seed, options.min_support) == (
        0.95,
        5,
        10_000,
        0,
        0.01,
    )
