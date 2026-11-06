# Roadmap

This file lists the planned releases of EBaMl. Each planned milestone has an implementation plan in `docs/plans/`. An
agent or a contributor can use the plan as the full task description. The PR that completes a milestone sets its status
to "Done". It also deletes the plan, unless a later milestone uses the same plan.

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
| 0.1.0 | Custom model providers: `model=` on every function that calls an LLM. | None | | Done |
| 0.2.0 | Model kinds: graphs and descriptions of multiclass classifiers, and local result types for every kind. | None | | Done |
| 0.3.0 | Custom audiences: `Audience` is a base class, and a user writes an audience with its own rules. | None | | Done |
| 0.4.0 | EBM2MCP: tools to read an EBM and to explain one prediction. | None | 0.1.0, 0.2.0, 0.3.0 | Done |
| 0.5.0 | Local explanations of regressors. | [local-explanations-all-kinds](docs/plans/local-explanations-all-kinds.md) | 0.4.0 | Done |
| 0.6.0 | Local explanations of multiclass classifiers. | [local-explanations-all-kinds](docs/plans/local-explanations-all-kinds.md) | 0.5.0 | Planned |
| 0.7.0 | Model editing in Python, in `ebaml.edit`. | [model-editing](docs/plans/model-editing.md) | 0.4.0 | Planned |
| 0.8.0 | EBM2MCP: edit tools. | [mcp-edit-tools](docs/plans/mcp-edit-tools.md) | 0.7.0 | Planned |
| 0.9.0 | GAM Changer parity: business constraints, the effect of an edit on data, and `.gamchanger` files. | [model-editing](docs/plans/model-editing.md) | 0.8.0 | Planned |
| 0.10.0 | Editing of every model kind. Every feature works for every model kind. | [model-editing](docs/plans/model-editing.md) | 0.6.0, 0.9.0 | Planned |
| **1.0.0** | **Stable API.** No breaking change without a new MAJOR version. | [stable-api](docs/plans/stable-api.md) | 0.10.0 | Planned |

Interaction terms are not on the list yet. The graph descriptions skip them, and the local explanations do not accept a
model that has them. The README table "Supported models" shows the state of each feature. Full support needs a
two-dimensional graph text and new proofs of the local explanations.

## Versions

EBaMl follows [Semantic Versioning 2.0.0](https://semver.org/) strictly. The package has no users yet, so 1.0.0 comes
last: it freezes the API only after every feature of the table is on `main`.

- Each pull request with a user-visible change adds a changelog fragment. After the merge, a workflow releases it with
  a new version.
- Before 1.0.0, any release can break the API. Such a change is marked **BREAKING** in the changelog. A release between
  two milestones raises the PATCH number, and the release that completes a milestone takes the version in the table.
- From 1.0.0, a breaking change needs a new MAJOR version, after a deprecation of at least one MINOR version.

`.claude/rules/changelog-and-versioning.md` has the full rule.
