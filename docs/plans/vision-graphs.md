# Plan: graphs for vision models (0.14.0)

You are a senior Python engineer with experience in multimodal LLM APIs, data visualization and BAML. You will implement
one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, with
special care for `.claude/rules/semantic-contracts.md`, `.claude/rules/reuse.md`, `.claude/rules/solid.md` and
`.claude/rules/testing.md`, because the project's hooks enforce those rules and `prek run --all-files` must pass before
you finish.

<goal>
Today EBaMl sends each graph of an EBM to the LLM as text (`EBMGraph.to_text`), because a text model can read only text.
Many current models are vision-language models (VLMs), which can also read an image. A person reads the shape of a graph
(a jump, a plateau, a trend) faster from a picture than from a table of bins, and a VLM may too. After this milestone,
the caller chooses how each graph reaches the model, with one option and three modes:

- **Text mode** sends the graph text only. It is the default, and it works with every model.
- **Vision mode** sends an image of the graph only.
- **Mixed mode** sends the graph text and the image of the same graph together.

A model without vision support must fail in a safe way. The caller chooses between two behaviors: the call falls back to
text mode and emits a warning that the caller can catch or filter, or the call raises one specific exception that the
caller can catch. A request never fails with a raw provider error that the user cannot interpret, and a call never
describes a graph that the model did not receive.
</goal>

<decisions>
The user made these choices. Do not reopen them.

1. The milestone lands before 1.0.0, so 1.0.0 freezes the option names that this milestone settles.
2. There are exactly three modes: text, vision and mixed. Text stays the default, so a call with no new option sends the
   same requests as today.
3. A model without vision support either falls back to text mode with a warning, or raises a specific exception that
   derives from `LLMRequestError`. The caller picks the behavior. Both behaviors exist in this milestone.
</decisions>

<context>
These facts were true when the plan was written. Check each one again before you act on it.

- `src/ebaml/graphs/ebm_graph.py` extracts an `EBMGraph`, and `graph_text.py` turns it into text. `token_budget.py`
  estimates the tokens of the text, and `to_text` simplifies the graph when the text is over budget.
- `scripts/docs/figures.py` draws the graph of a continuous feature of a binary classifier as an SVG document
  (`graph_svg`) for the docs site. It is the only renderer in the repository. It lives outside `src/`, and it covers one
  model kind and one feature type.
- The BAML toolchain (0.25) has a built-in `image` type (`baml.media.Image`, run `baml describe image`). It builds an
  image from base64 content with a MIME type (`image.from_base64`). Check with `baml describe` how a prompt of the
  toolchain puts an image into a message. Do not guess the syntax from older BAML docs.
- The providers that `client_builders` in `baml_src/model_spec.baml` supports accept PNG and JPEG images. OpenAI,
  Anthropic and Gemini do not accept SVG. So the image that crosses the boundary is a PNG, and SVG serves only the docs.
- `baml_src/talk_to_ebm.baml` holds the three-turn chain (`GraphFirstPass`, `GraphSurprises`, `GraphFinalDescription`),
  the summary and `describe_ebm`. `llm/descriptors.py` is the only module that imports `baml_sdk`.
- `ModelOptions` (`llm/options.py`) and `ModelSpec` (`model_spec.baml`) know nothing about the modalities of a model.
  The default model `openrouter/free` picks a model for each request, so one call can reach a model with vision and the
  next one a model without it. OpenRouter lists the input modalities of each model in its models API
  (`architecture.input_modalities`). Do not add a network call to the package to read it without asking the user.
- `ebaml.llm.provider_errors` maps the provider failures that a user can fix to `LLMRequestError` subclasses with fix
  steps. A provider that cannot read an image usually rejects the request with a 4xx status and a message about image
  input. The exact status and wording differ between providers. Collect real examples with the live tests before you
  write the matcher.
- The retry policy retries transient failures 3 times and does not retry a rejected request. A rejection for missing
  vision support must not be retried.
- Local explanations (`explain_instance`) send facts, not graphs. They are out of scope.
- Interaction terms are skipped and named today (contract 4). They stay skipped in every mode.
</context>

<design>
Settle these points in the first pull request. Ask the user only where the text says so.

- **The option.** Add an enum `GraphMode` with `TEXT`, `VISION` and `MIXED`, and a field
  `mode: GraphMode | str = GraphMode.TEXT` of `GraphOptions`. The mode is a choice about how a graph is presented, as
  `feature_format` is, so it belongs there. `describe_graph`, `describe_ebm` and the MCP tools that describe a graph read
  it. Export `GraphMode` from `ebaml`.
- **The fallback policy.** Add an enum `VisionFallback` with `TEXT` (fall back to text mode and warn) and `RAISE`, and a
  field `vision_fallback: VisionFallback | str = VisionFallback.RAISE` of `GraphOptions`. The default raises, because a
  caller who asked for an image must not get a text description in silence, and a warning is easy to miss in a script.
  The policy applies only when the mode is `VISION` or `MIXED`. Show the default to the user in the first pull request.
- **The exception.** Add `VisionUnsupportedError(LLMRequestError)` to `ebaml.core.exceptions`, and export it. Its message
  names the provider and the model, quotes the provider (as `provider_errors` does), and lists the fix steps: pick a model
  with image input, pass `GraphOptions(mode="text")`, or pass `vision_fallback="text"`. Add its row to `BUILTIN_BASE` of
  `tests/api/test_exceptions.py` and to the error table of `docs/usage.md`.
- **The warning.** Add `VisionFallbackWarning(UserWarning)` to `ebaml.core.exceptions`, and export it. Emit it with
  `warnings.warn`, because a caller filters a warning by its class or turns it into an error with
  `warnings.simplefilter("error", VisionFallbackWarning)`. Log a structlog event `vision_fallback` with the same fields too.
  Emit the warning once for each call, not once for each graph.
- **How EBaMl knows that a model cannot see.** Two sources, in this order:
  1. The caller declares it: add a field `vision: bool | None = None` to `ModelOptions`. `False` means that the model has
     no image input, so EBaMl applies the fallback policy before it sends any request. `True` and `None` send the image.
     Do not add a table of model names to the package: it goes stale with each model release.
  2. The provider rejects the image: `provider_errors` maps the rejection to `VisionUnsupportedError`. With
     `vision_fallback="text"`, the caller of the chain catches it and runs the same graph again in text mode.
  If the user wants a check of the OpenRouter models API too, add it as a third source only after the user agrees.
- **One answer for one call.** In `describe_ebm`, the first `VisionUnsupportedError` decides for the whole call. The
  graphs that wait in the thread pool switch to text mode (or stop, with `RAISE`) and do not each send a rejected request.
  The summary then says nothing that only an image showed. A test counts the requests of the fake server.
- **One graph, two views.** The image and the text of one graph come from the same `EBMGraph` object, so that they agree
  bin for bin (contract 5). In vision mode, no text is sent, so the text token budget does not apply, and the image is
  drawn from the extracted graph with no simplification. In mixed mode, the text budget simplifies the graph first, and
  the image is drawn from the simplified graph, so the model never sees two graphs that disagree.
- **The renderer.** Move the drawing code of `scripts/docs/figures.py` into a new module `src/ebaml/graphs/graph_image.py`,
  and make `figures.py` call it, so that the docs and the model see one renderer (`reuse.md`). Search the registry first,
  and register the renderer in `architecture/abstractions.yml`. The module returns bytes in memory and writes no file,
  because `graphs` has no I/O. It covers every model kind and feature type that the text covers: a step line with the
  confidence band for a continuous feature, bars with error bars for a categorical feature, one line or one group of bars
  for each class of a multiclass classifier with a legend, and the units of the target for a regressor. It draws the
  title, the axis names, the feature name and the class names from the graph, so that the image carries the same labels
  as the text. Keep the size and the colors fixed, so that the same graph gives the same bytes from run to run.
- **The PNG library. Ask the user.** The docs need SVG, and the providers need PNG. Show the user two or three options
  with their cost in install size and licence, for example `matplotlib` (wide use, large), or a small SVG rasterizer.
  Recommend one optional extra `vision`, so that a text-only user installs nothing more. If the extra is missing, a call
  in vision or mixed mode raises an error whose message names `pip install "ebaml[vision]"`, before any request.
- **The prompts.** Add a BAML class or parameters that carry the image next to the graph text, and keep one chain for all
  modes: the chain takes the graph text as optional and the image as optional, and a test rejects a call with neither.
  Write the prompts to Anthropic's guide (`writing-style.md`):
  - Vision mode: put the image first, then tell the model what the axes mean and that the band is the confidence interval
    at the stated level. Ask the model to read values from the axes as estimates, because a picture gives no exact value.
  - Mixed mode: tell the model that the text holds the exact values and the image shows the same graph, so it takes each
    number from the text and uses the image for the shape. This keeps the numbers of a description exact (contract 3).
  - Every mode names the omitted terms as today (contract 4).
- **The MCP server.** The agent of the MCP server is often a VLM. Add a tool `graph_image` that returns the image of one
  graph as MCP image content, so that the agent looks at the graph itself. Add `mode` and `vision_fallback` to the tools
  that call an LLM to describe a graph. Update the tool snapshot and `tests/api/test_mcp_tools.py`.
</design>

<steps>
Split the work into pull requests that pass every check alone. A suitable order:

1. **The renderer.** Add `graphs/graph_image.py` with SVG output for every model kind and feature type, move the code of
   `scripts/docs/figures.py` into it, and add PNG output behind the extra that the user chose. Unit tests: each model kind
   and feature type renders; the labels in the SVG equal the names in the graph; the same graph gives the same bytes; the
   points of the line equal the scores of the graph. No fragment yet if no public name changes.
2. **The options and the errors.** Add `GraphMode`, `VisionFallback`, `ModelOptions.vision`, `VisionUnsupportedError`
   and `VisionFallbackWarning`, with their exports, the API snapshot and the exception table. Validate the strings of the
   enums at the edge, as `normalization` does. Text mode sends the same requests as today, and a test shows it.
3. **The BAML side.** Add the image to the chain and the prompts. Offline BAML tests in
   `baml_src/tests/talk_to_ebm_test.baml`: the vision request carries one image and no graph text; the mixed request
   carries both; the text request carries no image; every mode names the omitted terms. Add the new crossings to `CALLED`
   in `tests/contract/test_sdk_calls.py`. Run `baml generate`, `baml fmt`, `baml check`, the offline tests and the live
   tests.
4. **The fallback.** Add the matcher to `provider_errors` with real rejection texts from the live tests, and the fallback
   in `llm/descriptors.py`. Integration tests with the local server (`fake_llm`), one reason to fail each:
   - The request of vision mode carries an image part in the wire format of `openai-generic`.
   - A rejection for image input raises `VisionUnsupportedError` with its fix steps, after one request (no retry).
   - With `vision_fallback="text"`, the same rejection gives a description, one `VisionFallbackWarning`, and a second
     request with the graph text and no image.
   - `ModelOptions(vision=False)` sends no image at all: with `RAISE` it sends no request, and with `TEXT` it sends text.
   - In `describe_ebm`, one rejection switches the whole call: count the requests.
   - A contract test checks that the image part matches what each provider of `client_builders` documents, through the
     request that the local server records.
5. **The MCP tools.** Add `graph_image` and the new arguments, with the snapshot and the API tests.
6. **The benchmark.** Add `--mode text|vision|mixed` to `benchmark/run.py`, and run the TalkToEBM tasks in each mode on
   one vision model. Report the accuracy of each mode in `docs/benchmark.md` with the model and the date. Do not claim in
   the docs that one mode is better without these numbers.
7. **Docs and notebook.** Add a section "Text, vision and mixed modes" to `docs/usage.md` with one example for each mode
   and one for each fallback behavior, a row for vision support in the table of `docs/providers.md`, and replace the limit
   about text-only graphs in `docs/providers.md`. Add a marimo notebook `notebooks/vision.py` (`notebooks.md`): it shows
   the image of each model kind with no key, then one section for each mode behind a button, then the two fallback
   behaviors with a text-only model. Register it in `scripts/docs/build.py`, add `docs/vision-notebook.md`, the nav entry
   in `zensical.toml` and a row in `docs/notebooks.md`. Update the README table "Supported models" if the modes need a row.
</steps>

<evidence>
For each contract that the change touches, name the guard and add a test (`semantic-contracts.md`):

| Contract | Guard to add or extend |
| --- | --- |
| 3 | Offline BAML tests: the vision prompt calls values read from the image estimates, and the mixed prompt takes numbers from the text only. |
| 4 | Extend `test_each_entry_point_names_or_rejects_a_model_with_an_interaction` to each mode, and the MCP twin to `graph_image`. |
| 5 | A unit test that the drawn points, the band and the labels of the image equal the scores, bounds and names of the `EBMGraph` that the text renders, for each model kind. |

Also guard the safety promise: a test for each path where a model without vision could receive an image, and a test that
no path describes a graph without sending it. Report each guarantee that no test shows.
</evidence>

<release>
This PR completes the milestone: it sets `version = "0.14.0"` in `pyproject.toml` and runs `uv lock`. Earlier pull
requests of the milestone each add their own fragment. The fragments list `GraphMode`, `VisionFallback`,
`ModelOptions.vision`, the new exception and warning, the extra and the MCP tool under `Added`. Mark the 0.14.0 row of
`ROADMAP.md` as done, and delete this plan.
</release>

<done>
- `describe_graph`, `describe_ebm` and the MCP tools describe a graph in text, vision or mixed mode, for every model kind.
- A model without vision support either falls back to text mode with one `VisionFallbackWarning`, or raises
  `VisionUnsupportedError`, as the caller chose, and no rejection is retried.
- The image and the text of a graph agree bin for bin, and every mode names the omitted terms.
- The benchmark reports the accuracy of each mode on one vision model.
- `prek run --all-files`, `uv run pytest`, the offline and live BAML tests, mutation testing of the changed lines and the
  strict docs build pass.
</done>
