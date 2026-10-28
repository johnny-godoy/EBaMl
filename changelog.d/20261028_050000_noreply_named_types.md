### Changed

- The type hints name the real types instead of `object`, so pyrefly and beartype check them: class labels are `ClassLabel`, the values of an instance are `FeatureValue`, and `Contribution.value` and `Change.from_value` are `str`. `ebaml.core.custom_types` defines the aliases.
