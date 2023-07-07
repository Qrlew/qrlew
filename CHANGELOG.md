# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- `filter` by `Expr` in `Schema` and `Field`
- filter in Relation builder
- Used `filter` field in `Map` when computing the schema
- Show `LIMIT` in the relation graph [MR49](https://github.com/Qrlew/qrlew/pull/49)
- Include `LIMIT` in the query when the derived from a Relation with a Map with a limit. [MR49](https://github.com/Qrlew/qrlew/pull/49)

### Added
- filter_iter in Relation builder
- Conversion for `Case` expression [MR1](https://github.com/Qrlew/qrlew/pull/1)
- Computation of the norm
- Add `clipped_sum` transform
- poisson_sampling transform [MR46](https://github.com/Qrlew/qrlew/pull/46)
- Added `filter` method in `Map` builder and `filter` transform [MR43](https://github.com/Qrlew/qrlew/pull/43)
- Map size propagation now takes into account `limit`. [MR49](https://github.com/Qrlew/qrlew/pull/49)
- Implement `IN` operator [MR50](https://github.com/Qrlew/qrlew/pull/50)
- Add methods for filtering fields in `Realtion`[MR51](https://github.com/Qrlew/qrlew/pull/51)

## [0.1.2] - 2023-06-01
### Added
- Md5 and Concat functions
- Protection operation
- Set operations support
- String functions `POSITION`, `LOWER`, `UPPER` and `CHAR_LENGTH` [PR9](https://github.com/Qrlew/qrlew/pull/9)
- Optional handling in or [PR6](https://github.com/Qrlew/qrlew/pull/6)

## [0.1.1] - 2023-05-26
### Changed
- Made sqlite optional

## [0.1.0] - 2023-05-25
### Added
- First commit open source

### Fixed
### Changed
### Removed
