### Added

- The MCP tool `check_constraints` finds the business constraints that the model breaks. Each violation holds the edit
  that repairs it, in the form that `preview_edit` and `apply_edit` take.

### Fixed

- The descriptions of the MCP edit tools give the real default of each field. They showed "whole graph" as the
  default of the bounds of `ValueInterval`.
