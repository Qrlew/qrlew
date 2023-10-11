use std::{
    ops::Deref,
    collections::HashMap, hash::Hash,
};

// A few utility objects
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct Step {
    pub referring_id: String,
    pub referred_relation: String,
    pub referred_id: String,
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

/// A path to a field
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct Path(pub Vec<Step>);

impl Deref for Path {
    type Target = Vec<Step>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO test this
impl<'a> FromIterator<(&'a str, &'a str, &'a str)> for Path {
    fn from_iter<T: IntoIterator<Item = (&'a str, &'a str, &'a str)>>(iter: T) -> Self {
        Path(
            iter.into_iter()
                .map(|(referring_id, referred_relation, referred_id)| Step::from((referring_id, referred_relation, referred_id)))
                .collect(),
        )
    }
}

// TODO test this
impl IntoIterator for Path {
    type Item = Step;
    type IntoIter = <Vec<Step> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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

/// A path to a field
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct FieldPath(pub Vec<ReferredField>);

impl FieldPath {
    pub fn from_path(
        path: Path,
        referred_field: String,
        referred_field_name: String,
    ) -> Self {
        let mut field_path = FieldPath(vec![]);
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in path {
            if let Some(last_step) = &mut last_step {
                field_path.0.push(ReferredField {
                    referring_id: last_step.referring_id.clone(),
                    referred_relation: last_step.referred_relation.clone(),
                    referred_id: last_step.referred_id.clone(),
                    referred_field: step.referring_id.clone(),
                    referred_field_name: referred_field_name.clone(),
                });
                *last_step = Step {
                    referring_id: referred_field_name.clone(),
                    referred_relation: step.referred_relation,
                    referred_id: step.referred_id,
                };
            } else {
                last_step = Some(step);
            }
        }
        if let Some(last_step) = last_step {
            field_path.0.push(ReferredField {
                referring_id: last_step.referring_id.clone(),
                referred_relation: last_step.referred_relation.clone(),
                referred_id: last_step.referred_id.clone(),
                referred_field,
                referred_field_name: referred_field_name.clone(),
            });
        }
        field_path
    }
}

impl Deref for FieldPath {
    type Target = Vec<ReferredField>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for FieldPath {
    type Item = ReferredField;
    type IntoIter = <Vec<ReferredField> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl From<(Vec<(&str, &str, &str)>, &str, &str)> for FieldPath {
    fn from((path, referred_field, referred_field_name): (Vec<(&str, &str, &str)>, &str, &str)) -> Self {
        FieldPath::from_path(Path::from_iter(path), referred_field.into(), referred_field_name.into())
    }
}

/// Associate a PEID to each table
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ProtectedEntity(pub HashMap<String, FieldPath>);

impl From<Vec<(&str, Vec<(&str, &str, &str)>, &str, &str)>> for ProtectedEntity {
    fn from(value: Vec<(&str, Vec<(&str, &str, &str)>, &str, &str)>) -> Self {
        let mut result = HashMap::new();
        for (table, protection, referred_field, referred_field_name) in value {
            result.insert(table.into(), FieldPath::from_path(Path::from_iter(protection), referred_field.into(), referred_field_name.into()));
        }
        ProtectedEntity(result)
    }
}

impl<'a> From<&'a ProtectedEntity> for Vec<(&'a str, Vec<(&'a str, &'a str, &'a str)>, &'a str, &'a str)> {
    fn from(value: &'a ProtectedEntity) -> Self {
        value.0.into_iter().map(|(table, field_path)| {
            let mut current_referred_field = &ReferredField::default();
            let mut path = vec![];
            for referred_field in &field_path.0 {
                current_referred_field = referred_field;
                path.push((referred_field.referring_id.as_str(), referred_field.referred_relation.as_str(), referred_field.referred_id.as_str()))
            }
            (table.as_str(), path, current_referred_field.referred_field.as_str(), current_referred_field.referred_field_name.as_str())
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
    // Add some tests
}