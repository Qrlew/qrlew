# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `Relation::Values` for supporting fixed values
- Add multiplicity module [MR68](https://github.com/Qrlew/qrlew/pull/68)
- sampling_without_replacements [MR68](https://github.com/Qrlew/qrlew/pull/68)
- Implemented utils for joins [MR62](https://github.com/Qrlew/qrlew/pull/62)

## [0.1.10] - 2023-07-24
### Changed
- Updated sqlparser to "0.36.1"
- Updated SQL -> Relation and Relation -> SQL
### Add
- Add a path to Tables to accomodate postgres schemas

## [0.1.10] - 2023-07-24
### Changed
- simplify intervals after union and intersection. [MR68](https://github.com/Qrlew/qrlew/pull/68)
- remove limit from poisson_sampling Relation transform [MR68](https://github.com/Qrlew/qrlew/pull/68)

## [0.1.9] - 2023-07-17
### Changed
- Deactivated display dot for integration tests
## [0.1.8] - 2023-07-17
### Fixed
- Fixed sqlparser version

## [0.1.7] - 2023-07-17
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
- Propagate fine grained `DataType` when `Expr::InList` in `Relation::Map` filter field [MR53](https://github.com/Qrlew/qrlew/pull/53)
- Add methods for filtering fields in `Realtion`[MR51](https://github.com/Qrlew/qrlew/pull/51)
- Implement `distinct_aggregates` transform that build `Relation` containing aggregates with the `DISTINCT` keyword [MR57](https://github.com/Qrlew/qrlew/pull/57)
- `Reduce::tau_thresholded_values` [MR60](https://github.com/Qrlew/qrlew/pull/60)
- Add `Reduce::protect_grouping_keys` [MR60](https://github.com/Qrlew/qrlew/pull/60)

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
