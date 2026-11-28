### Added

- `DescribeEBMRequest(render_timing=...)` chooses when `describe_ebm` renders its graphs. `RenderTiming.BEFORE_REQUESTS`, the default, renders every graph before the first request, so a graph that cannot fit the token budget fails before the LLM is paid. `RenderTiming.WITH_REQUESTS` renders each graph while the other requests wait for the LLM, which is 8 % faster on a model with 100 features and 1024 bins when each reply takes 100 ms.

### Changed

- The graph text is 40 to 58 % faster on every model of the performance benchmark, for example 1.31 s to 0.58 s for the 100 graphs of a model with 1024 bins, so `describe_ebm`, `describe_graph` and the graph tools of the MCP server wait less. Each text stays the same.
- `edit_effect` predicts each set of rows once for each model: 0.47 s to 0.22 s for an edit of a model with 100 features, and 69 ms to 49 ms for the `get_edit_effect` tool. The numbers stay the same.
- `explain_instance_async` computes the facts, and `describe_ebm_async` renders the graphs, in a thread, so the event loop serves other work meanwhile. The facts blocked the loop for up to 0.6 s on a model with 500 features.
- `import ebaml` loads skops only at the first save, which is about 0.25 s less at each start of a process.
