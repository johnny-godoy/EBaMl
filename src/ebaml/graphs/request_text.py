"""The text of the messages in the body of an LLM request.

The tokenizer of a provider counts the prompt, not the JSON around it. The body of a request holds the model name, the
sampling parameters and the structure of the messages besides the text. So the token learner reads the text of the
messages only. Providers shape the body differently (OpenAI, Anthropic and Gemini differ), so the function collects
the text under the keys that carry it. It needs no ``baml_sdk``.
"""

from collections.abc import Mapping, Sequence

import orjson

from ebaml.core.custom_types import JSONValue

# The keys whose value is text of the prompt, whatever the provider: `content` and `system` (OpenAI and Anthropic),
# `text` (the parts of OpenAI, Anthropic and Gemini) and `instructions` (the responses API of OpenAI).
_TEXT_KEYS = frozenset({"content", "text", "system", "instructions"})
# Keys below a text key that label a part and add no text: {"type": "text", "text": "..."}
_LABEL_KEYS = frozenset({"type", "role"})


def _collect(node: JSONValue, *, in_text: bool, found: list[str]) -> None:
    """Add the strings below a text key to ``found``, in the order of the body.

    Parameters
    ----------
    node: JSONValue
        The value to read.
    in_text: bool
        Whether a text key lies above the value.
    found: list[str]
        The strings found so far. The function appends to it.

    """
    if isinstance(node, str):  # nosemgrep: ebaml-polymorphism-not-type-branches -- JSON has no fixed shape
        if in_text:
            found.append(node)
    elif isinstance(node, Mapping):  # nosemgrep: ebaml-polymorphism-not-type-branches -- JSON has no fixed shape
        for key, value in node.items():
            if not (in_text and key in _LABEL_KEYS):
                _collect(value, in_text=in_text or key in _TEXT_KEYS, found=found)
    elif isinstance(node, Sequence):
        for value in node:
            _collect(value, in_text=in_text, found=found)


def message_text(body: str) -> str:
    """Collect the text of the messages from the body of a request, one part on each line.

    Parameters
    ----------
    body: str
        The JSON text of the request body.

    Returns
    -------
    str
        The text of the messages. It is empty when the body is not JSON or holds no text.

    Examples
    --------
    >>> message_text('{"model": "m", "messages": [{"role": "user", "content": "hello there"}]}')
    'hello there'
    >>> body = '{"system": "be brief", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}'
    >>> print(message_text(body))
    be brief
    hi

    """
    try:
        payload: JSONValue = orjson.loads(body)
    except orjson.JSONDecodeError:
        return ""
    found: list[str] = []
    _collect(payload, in_text=False, found=found)
    return "\n".join(found)
