### Changed

- The graph text of a model with many bins is 20 to 35 % faster, for example 633 ms to 446 ms for the 100 graphs of a model with 1024 bins, so `describe_ebm`, `describe_graph` and the graph tools of the MCP server wait less. Each text stays the same.
- `edit_effect` scores the rows once for each model: an edit of a model with 100 features and its effect take 153 ms instead of 272 ms. The numbers stay the same.
- `import ebaml` loads OR-Tools only at the first MIP search, which is about 0.1 s less at each start of a process that computes no local explanation.
