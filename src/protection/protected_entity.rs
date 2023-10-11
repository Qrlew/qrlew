use std::{
    ops::Deref,
    collections::HashMap, hash::Hash, fmt::Display,
};
use colored::Colorize;
use itertools::Itertools;

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
        write!(f, "{} {} {}.{}", self.referring_id.blue(), "→".red(), self.referred_relation.blue(), self.referred_id.blue())
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
        (&value.referring_id, &value.referred_relation, &value.referred_id)
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
                .map(|(referring_id, referred_relation, referred_id)| Step::from((referring_id, referred_relation, referred_id)))
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
        write!(f, "{}", self.iter().join(format!(" {} ", "|".yellow()).as_str()))
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
    pub fn new(referring_id: String, referred_relation: String, referred_id: String, referred_field: String, referred_field_name: String) -> ReferredField {
        ReferredField {
            referring_id, referred_relation, referred_id, referred_field, referred_field_name,
        }
    }
}

impl Display for ReferredField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {} AS {}", Step::new(self.referring_id.clone(), self.referred_relation.clone(), self.referred_id.clone()), "→".yellow(), self.referred_field, self.referred_field_name)
    }
}

/// A path to a field
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct FieldPath {
    path: Path,
    referred_field: String,
    referred_field_name: String,
}

impl FieldPath {
    pub fn new(path: Path, referred_field: String, referred_field_name: String) -> FieldPath {
        FieldPath {
            path, referred_field, referred_field_name,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn referred_field(&self) -> &str {
        &self.referred_field
    }

    pub fn referred_field_name(&self) -> &str {
        &self.referred_field_name
    }
}


impl Display for FieldPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {} AS {}", self.path, "→".yellow(), self.referred_field, self.referred_field_name)
    }
}

impl From<(Vec<(&str, &str, &str)>, &str, &str)> for FieldPath {
    fn from((path, referred_field, referred_field_name): (Vec<(&str, &str, &str)>, &str, &str)) -> Self {
        FieldPath::new(Path::from_iter(path), referred_field.into(), referred_field_name.into())
    }
}

impl<'a> From<&'a FieldPath> for (Vec<(&'a str, &'a str, &'a str)>, &'a str, &'a str) {
    fn from(value: &'a FieldPath) -> Self {
        ((&value.path).into(), &value.referred_field, &value.referred_field_name)
    }
}

impl<'a> IntoIterator for FieldPath {
    type Item = ReferredField;
    type IntoIter = <Vec<ReferredField> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let mut field_path = vec![];
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in self.path {
            if let Some(last_step) = &mut last_step {
                field_path.push(ReferredField::new(
                    last_step.referring_id.clone(),
                    last_step.referred_relation.clone(),
                    last_step.referred_id.clone(),
                    step.referring_id.clone(),
                    self.referred_field_name.clone(),
                ));
                *last_step = Step::new(
                    self.referred_field_name.clone(),
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
                self.referred_field,
                self.referred_field_name,
            ));
        }
        field_path.into_iter()
    }
}



/// Associate a PEID to each table
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ProtectedEntity(HashMap<String, FieldPath>);

impl Deref for ProtectedEntity {
    type Target = HashMap<String, FieldPath>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ProtectedEntity> for HashMap<String, FieldPath> {
    fn from(value: ProtectedEntity) -> Self {
        value.0
    }
}

impl From<Vec<(&str, Vec<(&str, &str, &str)>, &str, &str)>> for ProtectedEntity {
    fn from(value: Vec<(&str, Vec<(&str, &str, &str)>, &str, &str)>) -> Self {
        let mut result = HashMap::new();
        for (table, protection, referred_field, referred_field_name) in value {
            result.insert(table.into(), FieldPath::new(Path::from_iter(protection), referred_field.into(), referred_field_name.into()));
        }
        ProtectedEntity(result)
    }
}

impl<'a> From<&'a ProtectedEntity> for Vec<(&'a str, Vec<(&'a str, &'a str, &'a str)>, &'a str, &'a str)> {
    fn from(value: &'a ProtectedEntity) -> Self {
        value.iter().map(|(table, field_path)| {
            (table.as_str(), field_path.path().into(), field_path.referred_field(), field_path.referred_field_name())
        }).collect()
    }
}

impl Hash for ProtectedEntity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let vec_representation: Vec<(&str, Vec<(&str, &str, &str)>, &str, &str)> = self.into();
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
        let field_path: FieldPath = (vec![
            ("order_id", "order_table", "id"),
            ("user_id", "user_table", "id"),
        ],
        "name", "peid").into();
        println!("{:#?}", field_path);
    }

    #[test]
    fn test_length_zero_field_path() {
        let field_path: FieldPath = (vec![],
        "name", "peid").into();
        println!("{:#?}", field_path);
    }
}