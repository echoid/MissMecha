# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2] - 2025-07-30

### Improvements
- MNAR Type 1 now supports `missing_rate`-based quantile masking.  
  Thanks to [@mahshidkhatiri](https://github.com/mahshidkhatiri) for the feedback!

### New Features
- Added support for user-defined missing mechanisms via the `custom_class` parameter.
  - Support both global and column-wise simulation.
  - Easy interface: pass a class with `fit` + `transform` methods.
  - See [custom mechanism demo notebook](https://your-docs-link/examples/custom_mechanism_demo.html)

---

## [0.1.1] - 2025-04-27

### Bug Fixes
- Fixed MAR Type 2 crash when `depend_on` is a single column.
- Better error messages for invalid `info` config.

## [0.1.0] - 2025-04-09

### Initial Release
- MCAR, MAR, MNAR mechanism simulation
- Missing pattern visualization
- Type-aware imputation evaluation
