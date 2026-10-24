"""Describe EBM graphs and models using LLMs.

The prompts and the chat conversations live in ``baml_src/talk_to_ebm.baml``; this module extracts the graphs from the
EBM, converts them to text, and calls the generated ``baml_sdk`` functions.
"""

import inspect
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import attrs
import structlog
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml.custom_types import Audience, ExplainableBoostingModel, coerce_enum
from ebaml.graphs import EBMGraph
from ebaml.local import LocalFacts, LocalOptions, compute_local_facts
from ebaml.local_text import facts_to_texts
from ebaml.retry import call_with_shrinking_graph

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

_LOCAL_OPTION_NAMES = frozenset(attrs.fields_dict(LocalOptions))
_PROMPT_OPTION_NAMES = ("expert_description", "graph_description", "dataset_description", "task_description")
_GRAPH_SENTENCES = 7


def _accepted_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict:
    """Select the entries of ``kwargs`` that are parameters of ``func``.

    Parameters
    ----------
    func: Callable[..., Any]
        The function to inspect.
    kwargs: Mapping[str, Any]
        The keyword arguments to filter.

    Returns
    -------
    dict:
        A dictionary containing only the entries of ``kwargs`` that are parameters of ``func``.

    """
    names = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in names}


def _feature_importances_to_text(ebm: ExplainableBoostingModel) -> str:
    """Convert the feature importances of an EBM to text.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM model.

    Returns
    -------
        str: Textual representation of the feature importances.

    """
    return "\n".join(
        f"{feature_name}: {ebm.term_importances()[feature_idx]:.2f}"
        for feature_idx, feature_name in enumerate(ebm.feature_names_in_)
    )


def _prompt_options(kwargs: Mapping[str, Any]) -> baml_sdk.GraphPromptOptions:
    """Build the BAML prompt options from the keyword arguments, falling back to the BAML defaults.

    Parameters
    ----------
    kwargs: Mapping[str, Any]
        The keyword arguments to filter.

    Returns
    -------
    baml_sdk.GraphPromptOptions
        The prompt options for the BAML LLM client.

    """
    options = baml_sdk.default_graph_options()
    overrides = {k: kwargs[k] for k in _PROMPT_OPTION_NAMES if kwargs.get(k) is not None}
    return baml_sdk.GraphPromptOptions(**{**options.model_dump(), **overrides})


def _graph_text(ebm: ExplainableBoostingModel, feature_index: int, kwargs: Mapping[str, Any]) -> str:
    """Convert an EBM graph to text.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM model.
    feature_index: int
        The index of the feature to convert to text.
    kwargs: Mapping[str, Any]
        Additional keyword arguments to pass to ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.

    Returns
    -------
    str
        The textual representation of the EBM graph.

    """
    graph = EBMGraph.from_ebm(ebm, feature_index, **_accepted_kwargs(EBMGraph.from_ebm, kwargs))
    # the BAML check is the default, and a caller can replace it
    return graph.to_text(**{"check_budget": baml_sdk.check_token_budget, **_accepted_kwargs(EBMGraph.to_text, kwargs)})


def _is_truncated(error: BamlError) -> bool:
    """Tell whether an LLM call failed because the reply hit the ``max_tokens`` limit.

    Parameters
    ----------
    error: BamlError
        The error raised by a ``baml_sdk`` call.

    Returns
    -------
    bool
        True for a ``FinishReasonError`` whose finish reason is ``MaxTokens``.

    """
    value = error.value
    return error.class_name == "ai.errors.FinishReasonError" and getattr(value, "finish_reason", None) == "MaxTokens"


def _log_usage(what: str, counted: baml_sdk.Counted) -> str:
    """Log the token usage reported by the LLM provider and return the text.

    Parameters
    ----------
    what: str
        A label identifying the call in the log message.
    counted: baml_sdk.Counted
        The LLM reply together with its token usage.

    Returns
    -------
    str
        The text of the reply.

    """
    usage = counted.usage
    logger.info("usage", what=what, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
    return counted.text


def _describe_graph_counted(
    ebm: ExplainableBoostingModel,
    feature_index: int,
    graph: str,
    num_sentences: int,
    kwargs: Mapping[str, Any],
) -> baml_sdk.Counted:
    """Describe one graph with the LLM, simplifying the graph further whenever the reply is cut off by ``max_tokens``.

    The function raises the ``BamlError`` of the LLM call if the call fails. A reply cut off by ``max_tokens`` is
    retried first, as ``ebaml.retry.call_with_shrinking_graph`` says.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature to describe.
    graph: str
        The text of the graph, as made by ``_graph_text``.
    num_sentences: int
        The desired number of sentences for the description.
    kwargs: Mapping[str, Any]
        Passed to ``EBMGraph.from_ebm``, ``EBMGraph.to_text``, and the prompt options.

    Returns
    -------
    baml_sdk.Counted
        The description with its token usage.

    """
    opts = _prompt_options(kwargs)
    return call_with_shrinking_graph(
        lambda text: baml_sdk.describe_graph_cot(graph=text, num_sentences=num_sentences, opts=opts),
        graph,
        error_type=BamlError,
        is_truncated=_is_truncated,
        estimate_tokens=lambda text: baml_sdk.estimate_tokens(text=text),
        rebuild=lambda budget: _graph_text(
            ebm,
            feature_index,
            {**kwargs, "max_tokens": budget, "overbudget_percent": 0},
        ),
        label=ebm.feature_names_in_[feature_index],
    )


def describe_graph(
    ebm: ExplainableBoostingModel,
    feature_index: int,
    num_sentences: int = 7,
    **kwargs: object,
) -> str:
    """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

    The LLM client is configured in ``baml_src/talk_to_ebm.baml``. The function accepts additional keyword arguments
    that are passed to ``EBMGraph.from_ebm``, ``EBMGraph.to_text``, and the prompt (``expert_description``,
    ``graph_description``, ``dataset_description``, ``task_description``). A reply cut off by ``max_tokens`` is retried
    with a simplified graph, up to 4 times.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature to describe.
    num_sentences: int, default=7
        The desired number of sentences for the description.
    **kwargs: object
        Passed to ``EBMGraph.from_ebm``, ``EBMGraph.to_text``, and the prompt options.

    Returns
    -------
    str
        The description of the graph.

    """
    graph = _graph_text(ebm, feature_index, kwargs)
    counted = _describe_graph_counted(ebm, feature_index, graph, num_sentences, kwargs)
    return _log_usage(f"describe_graph({ebm.feature_names_in_[feature_index]})", counted)


def describe_ebm(
    ebm: ExplainableBoostingModel,
    num_sentences: int = 30,
    max_concurrency: int = 8,
    **kwargs: object,
) -> str:
    """Ask the LLM to describe an EBM.

    Each feature graph is described like in ``describe_graph`` (concurrently, at most ``max_concurrency`` at a time,
    each retried with a simplified graph if the reply is cut off by ``max_tokens``), and then the LLM summarizes the
    descriptions. The LLM client is configured in ``baml_src/talk_to_ebm.baml``. The function accepts additional
    keyword arguments that are passed to ``EBMGraph.from_ebm``, ``EBMGraph.to_text``, and the prompts
    (``expert_description``, ``dataset_description``, ``graph_description``, ``task_description``).

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    num_sentences: int, default=30
        The desired number of sentences for the description.
    max_concurrency: int, default=8
        The maximum number of features described at the same time.
    **kwargs: object
        Passed to ``EBMGraph.from_ebm``, ``EBMGraph.to_text``, and the prompt options.

    Returns
    -------
    str
        The description of the EBM.

    """
    # Convert all objects to text before prompting the LLM the first time, so that errors in processing one of the
    # graphs surface before any LLM call is paid for.
    feature_importances = _feature_importances_to_text(ebm)
    feature_names = list(ebm.feature_names_in_)
    graphs = [_graph_text(ebm, feature_index, kwargs) for feature_index in range(len(feature_names))]

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        described = list(
            pool.map(
                lambda item: _describe_graph_counted(ebm, item[0], item[1], _GRAPH_SENTENCES, kwargs),
                enumerate(graphs),
            ),
        )
    graph_descriptions = "\n\n".join(
        f"{name}: {item.text}"
        for name, item in zip(
            feature_names,
            described,
            strict=True,  # pragma: no mutate -- one item per feature, so strict never raises
        )
    )
    summary = baml_sdk.summarize_ebm(
        feature_importances=feature_importances,
        graph_descriptions=graph_descriptions,
        num_sentences=num_sentences,
        opts=_prompt_options(kwargs),
    )
    usage = baml_sdk.TokenUsage(
        input_tokens=summary.usage.input_tokens + sum(item.usage.input_tokens for item in described),
        output_tokens=summary.usage.output_tokens + sum(item.usage.output_tokens for item in described),
    )
    return _log_usage("describe_ebm", baml_sdk.Counted(text=summary.text, usage=usage))


@attrs.frozen
class LocalExplanation:
    """The explanation of one prediction: three kinds of explanation in words, and the facts they come from."""

    counterfactual: str
    sufficient_reason: str
    completions: str
    facts: LocalFacts


def explain_instance(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, Any],
    global_explanation: str,
    audience: Audience | str = Audience.LAYPERSON,
    num_sentences: int = 5,
    **kwargs: object,
) -> LocalExplanation:
    """Explain one prediction of an EBM with three explanations, computed exactly and put in words by the LLM.

    The facts come from ``ebaml.local.compute_local_facts``: the minimum change that flips the prediction
    (counterfactual), a minimum set of inputs that fixes it (sufficient reason), and the share of completions of a
    partially known input that get the same prediction (count completions). The global explanation of the model, e.g.
    the output of ``describe_ebm``, is prepended to each prompt. The LLM client is configured in
    ``baml_src/local_explanations.baml``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM: a binary classifier without interactions.
    instance: Mapping[str, Any]
        The value of every feature, by feature name.
    global_explanation: str
        The explanation of the whole model, computed once.
    audience: Audience | str, default=Audience.LAYPERSON
        Who reads the explanation. A layperson gets no model numbers; a data scientist gets scores, probabilities and
        confidence intervals.
    num_sentences: int, default=5
        The most sentences of each explanation.
    **kwargs: object
        The fields of ``LocalOptions`` (``confidence_level``, ``max_steps``, ``n_samples``, ``seed``,
        ``min_support``) go to ``compute_local_facts``; the prompt options (``expert_description``,
        ``dataset_description``) go to the prompts.

    Returns
    -------
    LocalExplanation
        The three explanations and the facts they come from.

    """
    audience = coerce_enum(Audience, audience)
    facts = compute_local_facts(
        ebm,
        instance,
        LocalOptions(**{k: v for k, v in kwargs.items() if k in _LOCAL_OPTION_NAMES}),
    )
    texts = facts_to_texts(facts, audience)
    result = baml_sdk.explain_local(
        global_explanation=global_explanation,
        facts=baml_sdk.LocalFactTexts(**texts),
        audience=audience.value,
        num_sentences=num_sentences,
        opts=_prompt_options(kwargs),
    )
    logger.info(
        "usage",
        what="explain_instance",
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )
    return LocalExplanation(result.counterfactual, result.sufficient_reason, result.completions, facts)
