# Tests of the rules in ebaml.yml. A comment that starts with the word ruleid marks a line that must match.
# ruleid: ebaml-preferred-libraries
import json
# ruleid: ebaml-preferred-libraries
import logging
# ruleid: ebaml-preferred-libraries
from dataclasses import dataclass
# ok: ebaml-preferred-libraries
import structlog


# ruleid: ebaml-no-kwargs-bag
def bag(**options):
    return options


# ok: ebaml-no-kwargs-bag
def named(a, *, b=1):
    return a, b


def chain(x):
    # ruleid: ebaml-no-elif-chain-on-one-value
    if x == 1:
        return "a"
    elif x == 2:
        return "b"
    elif x == 3:
        return "c"
    return "d"


def two_cases(x):
    # ok: ebaml-no-elif-chain-on-one-value
    if x == 1:
        return "a"
    elif x == 2:
        return "b"
    return "c"


class Base:
    def method(self):
        # ruleid: ebaml-no-not-implemented-stub
        raise NotImplementedError


def reads(path):
    # ruleid: ebaml-domain-does-no-io
    return path.read_text()


def shows(x):
    # ruleid: ebaml-domain-does-no-io
    print(x)


# ruleid: ebaml-no-manual-beartype
@beartype
def decorated(x):
    return x


def bare_error(x):
    if x < 0:
        # ruleid: ebaml-no-bare-value-error
        raise ValueError("negative")
    if x > 9:
        # ruleid: ebaml-no-bare-value-error
        raise ValueError("large") from None
    # ruleid: ebaml-no-bare-value-error
    raise ValueError


def own_error(x):
    if x < 0:
        # ok: ebaml-no-bare-value-error
        raise OptionRangeError("negative")
    try:
        return int(x)
    except ValueError:
        # ok: ebaml-no-bare-value-error
        raise


def fakes(monkeypatch, counted):
    # ruleid: ebaml-sdk-fake-is-checked
    monkeypatch.setattr(baml_sdk, "summarize_ebm", lambda **_kw: counted)
    # ruleid: ebaml-sdk-fake-is-checked
    monkeypatch.setattr(baml_sdk, "summarize_ebm", fake_sdk("describe_graph_cot", returning(counted)))
    # ok: ebaml-sdk-fake-is-checked
    monkeypatch.setattr(baml_sdk, "summarize_ebm", fake_sdk("summarize_ebm", returning(counted)))
