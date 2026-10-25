# Roadmap

This file lists the planned releases of EBaMl. Each milestone has an implementation plan in `docs/plans/`. An agent or
a contributor can use the plan as the full task description. The PR that completes a milestone sets its status to
"Done".

## Goals

EBaMl will reach three goals:

1. **Any LLM.** A user chooses the model and the provider from Python. A user does not edit the package.
2. **Any EBM.** Every feature works for binary classifiers, multiclass classifiers and regressors.
3. **EBM2MCP.** An MCP server gives an agent an EBM to read, to explain locally and to edit to meet business
   requirements. The editing part replaces [GAM Changer](https://github.com/interpretml/gam-changer), which has had no
   release since 2023.

## Milestones

| Version | Milestone | Plan | Needs | Status |
| --- | --- | --- | --- | --- |
| 0.0.3 | Release of the changes that are on `main`. | None | | Done |
| 0.1.0 | Custom model providers: `model=` on every function that calls an LLM. | [custom-model-providers](docs/plans/custom-model-providers.md) | | Planned |
| 0.2.0 | Model kinds: graphs and descriptions of multiclass classifiers, and local result types for every kind. | [model-kinds](docs/plans/model-kinds.md) | | Planned |
| 0.3.0 | EBM2MCP: tools to read an EBM and to explain one prediction. | [ebm2mcp](docs/plans/ebm2mcp.md) | 0.1.0, 0.2.0 | Planned |
| **1.0.0** | **Stable API.** No breaking change without a new MAJOR version. | [stable-api](docs/plans/stable-api.md) | 0.3.0 | Planned |
| 1.1.0 | Local explanations of regressors. | [local-explanations-all-kinds](docs/plans/local-explanations-all-kinds.md) | 1.0.0 | Planned |
| 1.2.0 | Local explanations of multiclass classifiers. | [local-explanations-all-kinds](docs/plans/local-explanations-all-kinds.md) | 1.1.0 | Planned |
| 1.3.0 | Model editing in Python, in `ebaml.experimental`. | [model-editing](docs/plans/model-editing.md) | 1.0.0 | Planned |
| 1.4.0 | EBM2MCP: edit tools, as experimental tools. | [ebm2mcp](docs/plans/ebm2mcp.md) | 1.3.0 | Planned |
| 1.5.0 | GAM Changer parity: business constraints, the effect of an edit on data, and `.gamchanger` files. | [model-editing](docs/plans/model-editing.md) | 1.4.0 | Planned |
| 1.6.0 | Stable editing. Every feature works for every model kind. | [model-editing](docs/plans/model-editing.md) | 1.2.0, 1.5.0 | Planned |

0.1.0 and 0.2.0 do not depend on each other. Two contributors can do them at the same time.

Interaction terms are not on the list yet. The graph descriptions skip them, and the local explanations do not accept a
model that has them. The [model-kinds](docs/plans/model-kinds.md) plan says what full support needs.

## Versions

EBaMl follows [Semantic Versioning 2.0.0](https://semver.org/) strictly.

- Each pull request with a user-visible change is a release. It raises the version and adds its own heading to
  `CHANGELOG.md`.
- Before 1.0.0, any release can break the API. Such a change is marked **BREAKING** in the changelog. A release between
  two milestones raises the PATCH number, and the release that completes a milestone takes the version in the table.
- From 1.0.0, a breaking change needs a new MAJOR version, after a deprecation of at least one MINOR version. A
  milestone after 1.0.0 is a MINOR release. Its number in the table shows the order, and other MINOR releases can come
  between two milestones.
- `ebaml.experimental` and the MCP tools whose names start with `experimental_` are not part of the stable API. They
  can change in any release.

`.claude/rules/changelog-and-versioning.md` has the full rule.
