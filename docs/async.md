---
description: The async form of each function of EBaMl that calls an LLM, with concurrency, partial results, cancellation and timeouts.
---

# Async API

Each function of EBaMl that calls an LLM has two forms. The sync form returns when the LLM answers. The async form,
with the suffix `_async`, is a coroutine that awaits the LLM. Both forms take the same request and return the same
result.

| Sync form | Async form |
| --- | --- |
| `describe_graph` | `describe_graph_async` |
| `describe_ebm` | `describe_ebm_async` |
| `explain_instance` | `explain_instance_async` |
| `propose_difficulties` | `propose_difficulties_async` |

The [async notebook](async-notebook.md) explains several predictions at the same time with the async forms.

## Choose the form

- Use the sync form in a script, in a test, or in a notebook cell that waits for one answer.
- Use the async form in a program that runs an event loop: a web server, an agent framework, an MCP server. The loop
  then serves other work while the LLM answers.
- Use the async form to send several calls at the same time, for example to explain many predictions.

Both forms run one implementation of each call. The graph text, the retry after a cut-off reply, the vision fallback
and the errors are the same in both forms.

## One example for each function

Run a coroutine with `asyncio.run` in a script. In a program that runs an event loop, use `await` in place of
`asyncio.run`.

```python
import asyncio

from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
    describe_ebm_async,
    describe_graph_async,
    explain_instance_async,
    propose_difficulties_async,
)

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)
instance = X.iloc[0].to_dict()

graph_text = asyncio.run(describe_graph_async(DescribeGraphRequest(ebm, feature_index=0)))
description = asyncio.run(describe_ebm_async(DescribeEBMRequest(ebm)))
explanation = asyncio.run(explain_instance_async(ExplainInstanceRequest(ebm, instance, description.text)))
proposal = asyncio.run(propose_difficulties_async(ProposeDifficultiesRequest(ebm, instance)))
print(explanation.counterfactual)
```

## Several calls at the same time

Start the calls in one task group. The requests of the calls then wait for the LLM at the same time.

```python
async def explain_rows(rows: list[int]) -> list[str]:
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(
                explain_instance_async(ExplainInstanceRequest(ebm, X.iloc[row].to_dict(), description.text)),
            )
            for row in rows
        ]
    return [task.result().counterfactual for task in tasks]


counterfactuals = asyncio.run(explain_rows([0, 1, 2]))
```

A provider limits the rate of the requests. If you start many calls, limit them with an `asyncio.Semaphore`.

## Concurrency in describe_ebm

`describe_ebm` and `describe_ebm_async` describe the graphs of the features at the same time. `max_concurrency` sets
the most graphs in flight at one time, and the default is 8. The sync form describes the graphs in a pool of threads.
The async form describes them as tasks of one task group in the event loop.

```python
description = asyncio.run(describe_ebm_async(DescribeEBMRequest(ebm, max_concurrency=2, progress_bar=True)))
```

`progress_bar=True` shows one step for each feature and one step for the summary, in both forms.

## Partial results of describe_ebm

The request of one feature can fail while the other requests succeed, for example after a rate limit. Both forms then
describe the other features. `EBMDescription.failed_features` names each feature that failed, with its error, and the
summary says that it does not cover these features. A `features_failed` warning names them in the log.

```python
for failure in description.failed_features:
    print(f"{failure.feature}: {failure.error}")
```

An error has the class attribute `affects_every_request`. If it is true, every request of the call fails in the same
way, so `describe_ebm` stops at once and raises the error.

| Error | The call |
| --- | --- |
| `MissingAPIKeyError`, `APIKeyRejectedError` | Stops at once. |
| `ProviderUnreachableError` | Stops at once. |
| `VisionUnsupportedError` | Stops at once, after the vision fallback. |
| `RateLimitedError`, after the retries | Describes the other features. |
| `LLMReplyError`: a reply that stays cut off after the retries, or that does not parse | Describes the other features. |

If every feature fails, `describe_ebm` raises the error of the first feature, in the order of the features, and sends
no summary request. `describe_graph` describes one feature, so it raises its error.

## Cancellation and timeouts

Cancel the task of a call to stop it. The async form then sends no further request. The requests that are in flight
end on the provider, and the provider can count their tokens.

Set a time limit with `asyncio.timeout`. When the time ends, the call stops and `TimeoutError` is raised.

```python
async def describe_in_time() -> str | None:
    try:
        async with asyncio.timeout(120):
            return (await describe_ebm_async(DescribeEBMRequest(ebm))).text
    except TimeoutError:
        return None


text = asyncio.run(describe_in_time())
```

## The sync form inside a running event loop

Jupyter and marimo run an event loop for the cells. The sync form works there: it blocks the loop until the LLM
answers, and it gives the same result. In a cell that can use `await`, prefer the async form, so that the loop stays
free.

## Steps that block the event loop

The async form awaits the LLM requests only. The other steps run in the event loop, and the loop waits while they run:

- the extraction, the simplification and the text of each graph,
- the images of the vision and the mixed modes,
- the facts of a local explanation.

These steps take less than a second for a usual model. A model with many features, or a local explanation with a
large search, takes longer. If your server must answer during that time, run the whole call in a task of its own and
keep a second worker for the other requests.
