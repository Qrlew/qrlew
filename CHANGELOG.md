# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
- `CAST` function [MR188](https://github.com/Qrlew/qrlew/pull/188)

## [0.5.1] - 2023-11-19
## Added
- Added `Count(*)` by parsing it as `Count(1)`

### Changed
- Add a warning when the textbook gaussian mechanism is not really applied [#190](https://github.com/Qrlew/qrlew/issues/190)

## [0.5.0] - 2023-11-19
### Changed
- Breaking name change: Protected Entity -> Privacy Unit + PEP -> PUP [#189](https://github.com/Qrlew/qrlew/issues/189)

## [0.4.13] - 2023-11-14
## Added
- `TRIM` function [MR183](https://github.com/Qrlew/qrlew/pull/183)
- `SUBSTR` function [MR186](https://github.com/Qrlew/qrlew/pull/186)

## [0.4.12] - 2023-11-09
### Fixed
- Dp query should release all the possible values of the grouping keys. [MR180](https://github.com/Qrlew/qrlew/pull/180)
- `DataType`` propagation in joins: if their is not INNER or CROSS contraint, then the output `DataType`s must be optional [MR179](https://github.com/Qrlew/qrlew/pull/179)
### Added
- Implemented `Coalesce` [MR178](https://github.com/Qrlew/qrlew/pull/178)
- `TRIM` function [MR183](https://github.com/Qrlew/qrlew/pull/183)
### Changed
- If no tau-thresholding, the budget is transferred to the aggregations [MR180](https://github.com/Qrlew/qrlew/pull/180)
- Allow public tables [MR182](https://github.com/Qrlew/qrlew/pull/182)

## [0.4.11] - 2023-11-09
### Fixed
- Use a connection pool (R2D2) for multithreaded access to DB

## [0.4.10] - 2023-11-09
### Fixed
- Retries only 10 times

## [0.4.9] - 2023-11-09
### Fixed
- Do not split budget in half when the query has no GROUP BY. Use all the budget in the aggregations

## [0.4.8] - 2023-11-09
### Fixed
- Retries in postgres

## [0.4.7] - 2023-11-09
### Fixed
- DP can be SD

## [0.4.6] - 2023-10-27
### Fixed
- fixed clipped noise

## [0.4.5] - 2023-10-27
### Changed
- added clipped noise [MR171](https://github.com/Qrlew/qrlew/pull/171)

## [0.4.4] - 2023-10-27
### Fixed
- changed PEP compilation

## [0.4.3] - 2023-10-27
### Fixed
- added rewrite_as_pep [MR170](https://github.com/Qrlew/qrlew/pull/170)
- Updates sqlparser version

## [0.4.2] - 2023-10-27
### Fixed
- gaussian noise [MR169](https://github.com/Qrlew/qrlew/pull/169)

## [0.4.1] - 2023-10-24
### Added
- Differential privacy in rules [MR156](https://github.com/Qrlew/qrlew/pull/156)
- Method for dp rewritting: rewrite_with_differential_privacy [MR162](https://github.com/Qrlew/qrlew/pull/162)
- computation of the size in the case of join where the constraint is unique [MR163](https://github.com/Qrlew/qrlew/pull/163)
- - `rewriting_rul.RewriteVisitor` outputs a `RelationWithPrivateQuery` [MR157](https://github.com/Qrlew/qrlew/pull/157)
### Fixed
- differential privacy for the new formalism [MR155](https://github.com/Qrlew/qrlew/pull/155)
- bug in differential privacy [MR158](https://github.com/Qrlew/qrlew/pull/158)
### Changed
- `set rewrite_with_differential_privacy` as a Relation's method [MR166](https://github.com/Qrlew/qrlew/pull/166)

## [0.4.0] - 2023-10-19
### Added
- support for filtering datatypes by columns and values [MR138](https://github.com/Qrlew/qrlew/pull/138)
- Support for `HAVING` [MR141](https://github.com/Qrlew/qrlew/pull/141)
- `names` filed in `Join::Builder` [MR153](https://github.com/Qrlew/qrlew/pull/153)
- Added Budget split
- Added Score
### Fixed
- `filter` for `Map` and `Reduce` builders [MR137](https://github.com/Qrlew/qrlew/pull/137)
- `expr::Function::Pointwise` [MR140](https://github.com/Qrlew/qrlew/pull/140)
- protection for `Join` [MR147](https://github.com/Qrlew/qrlew/pull/147)
- Fixed the rest of protection

## [0.3.8] - 2023-09-29
### Changed
- `DPRelation` deref to `Relation`

## [0.3.7] - 2023-09-28
### Changed
- Objects in `Arc`s are `Sync + Send` for thread safety and matable objects are behind `Mutex`es

## [0.3.6] - 2023-09-28
### Changed
- All `Rc`s have been changed into `Arc`s for thread safety

## [0.3.5] - 2023-09-28
### Changed
- Implemented a Visitor for doing the DP compilation (`differential_privacy::dp_compile``). Its handles both aggregates and group by columns. [MR129](https://github.com/Qrlew/qrlew/pull/129)

### Fixed
- Order of relations in the Dot representation of `Relation::Join`[MR131](https://github.com/Qrlew/qrlew/pull/131)

## [0.3.4] - 2023-09-25
- Fixed examples

## [0.3.3] - 2023-09-25
### Changed
- Updated `sqlparser`

## [0.3.2] - 2023-09-25
### Added
- conversion DataType -> Value for Expr::Function [MR122](https://github.com/Qrlew/qrlew/pull/122)
- replace `BTreeSet` in methods `super_union` and `super_intersection` of `data_type::Struct` and `data_type::Union`
in order keep the inserting order (`BTreeSet` reorder keys by alphanumeric order) [MR125](https://github.com/Qrlew/qrlew/pull/125)
### Fixed
- in protection use `PEPRelation::try_from(..)` instead of `PEPRelation(..)` [MR124](https://github.com/Qrlew/qrlew/pull/124)
- dp_compile [MR127](https://github.com/Qrlew/qrlew/pull/127)
### Changed
- Use `PEPRelation` when protecting the grouping keys [MR126](https://github.com/Qrlew/qrlew/pull/126)
- Protection paths are given with vecs and not slices anymore

## [0.3.1] - 2023-09-16
### Fixed
- Fixed the dp compilation
- Datatype filtering for joins [MR121](https://github.com/Qrlew/qrlew/pull/121)

## [0.3.0] - 2023-09-14
### Changed
- Replaced `Expr::filter_column_data_type` by `DataType::filter`[MR104](https://github.com/Qrlew/qrlew/pull/104)
- Remove `Hierarchy::chain` and replace it by `with` when needed [MR113](https://github.com/Qrlew/qrlew/pull/113)
### Fixed
- `Union::is_subset_of` [MR106](https://github.com/Qrlew/qrlew/pull/106)
- fix reduce when the query has a group by and doesn't have aggregation functions [MR80](https://github.com/Qrlew/qrlew/pull/80)
### Added
- support for `expr::Function::Or` in `DataType::filter_by_function` [MR110](https://github.com/Qrlew/qrlew/pull/110)
- Can compile recursively
- Checks for possible values
- Can compile more than sums

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
