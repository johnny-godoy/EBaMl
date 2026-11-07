### Fixed

- A fresh `pip install ebaml` no longer installs a `baml-bridge` that rejects the generated `baml_sdk` with a version
  skew error. `ebaml` now requires `baml-bridge==0.23.0`, the version of the toolchain that generates the SDK.
