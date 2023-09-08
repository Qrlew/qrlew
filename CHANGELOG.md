# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2023-09-04
### Changed
- Internal code uses `Relation.name()` for table addressing but user facing functions may use `Table.path()` (ie sql addressing)
- Renamed bivariate_min and bivariate_max to least and greatest
- Cast to string before MD5 for protection
- Implemented `least` and `greatest` (support qualified and unqualified columns)[MR102](https://github.com/Qrlew/qrlew/pull/102)
### Fixed
- `And` for struct of structs [MR100](https://github.com/Qrlew/qrlew/pull/100)
### Added
- `Hierarchy::get_key_value` [MR103](https://github.com/Qrlew/qrlew/pull/103)

## [0.2.2] - 2023-08-29
### Changed
- module name: `multiplicity` -> `sampling_adjustments` [MR77](https://github.com/Qrlew/qrlew/pull/77)
- more coherent objects and function names inside `sampling_adjustments` [MR77](https://github.com/Qrlew/qrlew/pull/77)
- Updated sqlparser version
- Deactivate graphviz display by default
- Deactivate multiplicity testing by default
-
### Added
- In `sampling_adjustments` added differenciated sampling and adjustments [MR77](https://github.com/Qrlew/qrlew/pull/77)
- Updated sqlparser version
- Deactivate graphviz display by default
- Deactivate multiplicity testing by default
- Improved Index trait for `data_type::Value` and `DataType`[MR94](https://github.com/Qrlew/qrlew/pull/94)
- Implemented `hierarchy` method for `data_type::Value` and `DataType`[MR94](https://github.com/Qrlew/qrlew/pull/94)

## [0.2.1] - 2023-07-26
### Added
- join utils [MR72](https://github.com/Qrlew/qrlew/pull/72)
- `Relation::possible_values` and used it in thresholding [MR73](https://github.com/Qrlew/qrlew/pull/73)
- Error handling in transforms and diffrential_privacy [MR59](https://github.com/Qrlew/qrlew/pull/59)
- Support for filtering by expression i.e. a > 3 * 5 [MR81](https://github.com/Qrlew/qrlew/pull/81)
### Changed
- `Relation::filter_columns`
-  join utils [MR72](https://github.com/Qrlew/qrlew/pull/72)
-  Fixed table naming
-  Made tests for multiplicity optional to avaoid Memory errors in the CI [MR84](https://github.com/Qrlew/qrlew/pull/84)
### Fixed
- Injection Float -> Integer and DataType.super_intersection [MR84](https://github.com/Qrlew/qrlew/pull/84)

## [0.2.0] - 2023-07-25
### Added
- `Relation::Values` for supporting fixed values
- multiplicity module [MR68](https://github.com/Qrlew/qrlew/pull/68)
- `sampling_without_replacements` [MR68](https://github.com/Qrlew/qrlew/pull/68)

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
