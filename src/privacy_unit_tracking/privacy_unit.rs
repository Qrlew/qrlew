use colored::Colorize;
use itertools::Itertools;
use std::{fmt::Display, hash::Hash, ops::Deref};

pub const PRIVACY_PREFIX: &str = "_PRIVACY_";
pub const PRIVACY_COLUMNS: usize = 2;
pub const PRIVACY_UNIT: &str = "_PRIVACY_UNIT_";
pub const PRIVACY_UNIT_DEFAULT: &str = "_PRIVACY_UNIT_DEFAULT_";
pub const PRIVACY_UNIT_WEIGHT: &str = "_PRIVACY_UNIT_WEIGHT_";
pub const PRIVACY_UNIT_ROW: &str = "_PRIVACY_UNIT_ROW_";

// A few utility objects

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct Step {
    referring_id: String,
    referred_relation: String,
    referred_id: String,
}

impl Step {
    pub fn new(referring_id: String, referred_relation: String, referred_id: String) -> Step {
        Step {
            referring_id,
            referred_relation,
            referred_id,
        }
    }
}

impl Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {}.{}",
            self.referring_id.blue(),
            "→".red(),
            self.referred_relation.blue(),
            self.referred_id.blue()
        )
    }
}

impl From<(&str, &str, &str)> for Step {
    fn from((referring_id, referred_relation, referred_id): (&str, &str, &str)) -> Self {
        Step {
            referring_id: referring_id.into(),
            referred_relation: referred_relation.into(),
            referred_id: referred_id.into(),
        }
    }
}

impl<'a> From<&'a Step> for (&'a str, &'a str, &'a str) {
    fn from(value: &'a Step) -> Self {
        (
            &value.referring_id,
            &value.referred_relation,
            &value.referred_id,
        )
    }
}

/// A path to a field
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct Path(Vec<Step>);

impl Deref for Path {
    type Target = Vec<Step>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Path> for Vec<Step> {
    fn from(value: Path) -> Self {
        value.0
    }
}

impl<'a> From<&'a Path> for Vec<(&'a str, &'a str, &'a str)> {
    fn from(value: &'a Path) -> Self {
        value.0.iter().map(|step| step.into()).collect()
    }
}

impl<'a> FromIterator<(&'a str, &'a str, &'a str)> for Path {
    fn from_iter<T: IntoIterator<Item = (&'a str, &'a str, &'a str)>>(iter: T) -> Self {
        Path(
            iter.into_iter()
                .map(|(referring_id, referred_relation, referred_id)| {
                    Step::from((referring_id, referred_relation, referred_id))
                })
                .collect(),
        )
    }
}

impl IntoIterator for Path {
    type Item = Step;
    type IntoIter = <Vec<Step> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.iter().join(format!(" {} ", "|".yellow()).as_str())
        )
    }
}

/// A link to a relation and a field to keep with a new name
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct ReferredField {
    pub referring_id: String,
    pub referred_relation: String,
    pub referred_id: String,
    pub referred_field: String,
    pub referred_field_name: String,
}

impl ReferredField {
    pub fn new(
        referring_id: String,
        referred_relation: String,
        referred_id: String,
        referred_field: String,
        referred_field_name: String,
    ) -> ReferredField {
        ReferredField {
            referring_id,
            referred_relation,
            referred_id,
            referred_field,
            referred_field_name,
        }
    }
}

impl Display for ReferredField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {} AS {}",
            Step::new(
                self.referring_id.clone(),
                self.referred_relation.clone(),
                self.referred_id.clone()
            ),
            "→".yellow(),
            self.referred_field,
            self.referred_field_name
        )
    }
}

/// A path to a field
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct PrivacyUnitPath {
    path: Path,
    privacy_unit_field: String,
}

impl PrivacyUnitPath {
    pub fn new(path: Path, privacy_unit_field: String) -> PrivacyUnitPath {
        PrivacyUnitPath {
            path,
            privacy_unit_field,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn referred_field(&self) -> &str {
        &self.privacy_unit_field
    }

    pub fn privacy_unit() -> &'static str {
        PRIVACY_UNIT
    }

    pub fn privacy_unit_default() -> &'static str {
        PRIVACY_UNIT_DEFAULT
    }
}

impl Display for PrivacyUnitPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {} AS {}",
            self.path,
            "→".yellow(),
            self.privacy_unit_field,
            PrivacyUnitPath::privacy_unit(),
        )
    }
}

impl From<(Vec<(&str, &str, &str)>, &str)> for PrivacyUnitPath {
    fn from((path, referred_field): (Vec<(&str, &str, &str)>, &str)) -> Self {
        PrivacyUnitPath::new(Path::from_iter(path), referred_field.into())
    }
}

impl<'a> From<&'a PrivacyUnitPath> for (Vec<(&'a str, &'a str, &'a str)>, &'a str) {
    fn from(value: &'a PrivacyUnitPath) -> Self {
        ((&value.path).into(), &value.privacy_unit_field)
    }
}

impl<'a> IntoIterator for PrivacyUnitPath {
    type Item = ReferredField;
    type IntoIter = <Vec<ReferredField> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let mut field_path = vec![];
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in self.path {
            if let Some(last_step) = &mut last_step {
                field_path.push(ReferredField::new(
                    last_step.referring_id.to_string(),
                    last_step.referred_relation.to_string(),
                    last_step.referred_id.to_string(),
                    step.referring_id.to_string(),
                    PrivacyUnitPath::privacy_unit().to_string(),
                ));
                *last_step = Step::new(
                    PrivacyUnitPath::privacy_unit().to_string(),
                    step.referred_relation,
                    step.referred_id,
                );
            } else {
                last_step = Some(step);
            }
        }
        if let Some(last_step) = last_step {
            field_path.push(ReferredField::new(
                last_step.referring_id,
                last_step.referred_relation,
                last_step.referred_id,
                self.privacy_unit_field,
                PrivacyUnitPath::privacy_unit().to_string(),
            ));
        }
        field_path.into_iter()
    }
}

/// Associate a PID to each table
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PrivacyUnit {
    paths: Vec<(String, PrivacyUnitPath)>,
    pu_column: Option<String>,
    weight: Option<String>,
}

impl PrivacyUnit {
    pub fn new(
        paths: Vec<(String, PrivacyUnitPath)>,
        pu_column: Option<String>,
        weight: Option<String>,
    ) -> Self {
        PrivacyUnit {
            paths,
            pu_column,
            weight
        }
    }

    pub fn privacy_prefix() -> &'static str {
        PRIVACY_PREFIX
    }

    pub fn privacy_columns() -> usize {
        PRIVACY_COLUMNS
    }

    pub fn privacy_unit_row() -> &'static str {
        PRIVACY_UNIT_ROW
    }

    pub fn privacy_unit(&self) -> String {
        self.pu_column.clone().unwrap_or(PrivacyUnitPath::privacy_unit().to_string())
    }

    pub fn privacy_unit_default() -> &'static str {
        PrivacyUnitPath::privacy_unit()
    }

    pub fn privacy_unit_weight(&self) -> String {
        self.weight.clone().unwrap_or(PRIVACY_UNIT_WEIGHT.to_string())
    }

    pub fn privacy_unit_weight_default() -> &'static str {
        PRIVACY_UNIT_WEIGHT
    }

    pub fn with_pu_column_and_weight(self, pu_column: String, weight: String) -> Self {
        PrivacyUnit{paths: self.paths, pu_column: Some(pu_column), weight: Some(weight)}
    }
}

impl Display for PrivacyUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.iter()
                .map(|(table, field_path)| format!("{} {} {}", table, "→".cyan(), field_path))
                .join("\n")
        )
    }
}

impl Deref for PrivacyUnit {
    type Target = Vec<(String, PrivacyUnitPath)>;

    fn deref(&self) -> &Self::Target {
        &self.paths
    }
}

impl From<PrivacyUnit> for Vec<(String, PrivacyUnitPath)> {
    fn from(value: PrivacyUnit) -> Self {
        value.paths
    }
}

impl From<Vec<(&str, Vec<(&str, &str, &str)>, &str)>> for PrivacyUnit {
    fn from(value: Vec<(&str, Vec<(&str, &str, &str)>, &str)>) -> Self {
        let mut result = vec![];
        for (table, privacy_unit_tracking, referred_field) in value {
            result.push((
                table.into(),
                PrivacyUnitPath::new(
                    Path::from_iter(privacy_unit_tracking),
                    referred_field.into(),
                ),
            ));
        }
        PrivacyUnit::new(result, None, None)
    }
}

impl<'a> From<&'a PrivacyUnit> for Vec<(&'a str, Vec<(&'a str, &'a str, &'a str)>, &'a str)> {
    fn from(value: &'a PrivacyUnit) -> Self {
        value
            .iter()
            .map(|(table, field_path)| {
                (
                    table.as_str(),
                    field_path.path().into(),
                    field_path.referred_field(),
                )
            })
            .collect()
    }
}

impl Hash for PrivacyUnit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let vec_representation: Vec<(&str, Vec<(&str, &str, &str)>, &str)> = self.into();
        vec_representation.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        let step = Step::from(("order_id", "order_table", "id"));
        println!("{step}");
    }

    #[test]
    fn test_path() {
        let path = Path::from_iter(vec![
            ("order_id", "order_table", "id"),
            ("user_id", "user_table", "id"),
        ]);
        println!("{path}");
    }

    #[test]
    fn test_referred_field() {
        let referred_field = ReferredField::new(
            "order_id".into(),
            "order_table".into(),
            "id".into(),
            "name".into(),
            "peid".into(),
        );
        println!("{referred_field}");
    }

    // Add some tests
    #[test]
    fn test_field_path() {
        let field_path: PrivacyUnitPath = (
            vec![
                ("order_id", "order_table", "id"),
                ("user_id", "user_table", "id"),
            ],
            "name",
        )
            .into();
        println!("{}", field_path);
    }

    #[test]
    fn test_length_zero_field_path() {
        let field_path: PrivacyUnitPath = (vec![], "name").into();
        println!("{}", field_path);
    }

    // Add some tests
    #[test]
    fn test_privacy_unit() {
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
            ("product_table", vec![], PRIVACY_UNIT_ROW),
        ]);
        println!("{}", privacy_unit);
    }
}
