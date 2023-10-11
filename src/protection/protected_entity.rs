use std::{
    ops::Deref,
};

// A few utility objects
#[derive(Clone, Debug)]
pub struct Step<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
}

impl<'a> From<(&'a str, &'a str, &'a str)> for Step<'a> {
    fn from((referring_id, referred_relation, referred_id): (&'a str, &'a str, &'a str)) -> Self {
        Step {
            referring_id,
            referred_relation,
            referred_id,
        }
    }
}


#[derive(Clone, Debug)]
pub struct Path<'a>(pub Vec<Step<'a>>);

impl<'a> Deref for Path<'a> {
    type Target = Vec<Step<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromIterator<&'a (&'a str, &'a str, &'a str)> for Path<'a> {
    fn from_iter<T: IntoIterator<Item = &'a (&'a str, &'a str, &'a str)>>(iter: T) -> Self {
        Path(
            iter.into_iter()
                .map(|(referring_id, referred_relation, referred_id)| Step {
                    referring_id,
                    referred_relation,
                    referred_id,
                })
                .collect(),
        )
    }
}

impl<'a> IntoIterator for Path<'a> {
    type Item = Step<'a>;
    type IntoIter = <Vec<Step<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A link to a relation and a field to keep with a new name
#[derive(Clone, Debug)]
pub struct ReferredField<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
    pub referred_field: &'a str,
    pub referred_field_name: &'a str,
}

/// A path to a field
#[derive(Clone, Debug)]
pub struct FieldPath<'a>(pub Vec<ReferredField<'a>>);

impl<'a> FieldPath<'a> {
    pub fn from_path(
        path: Path<'a>,
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Self {
        let mut field_path = FieldPath(vec![]);
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in path {
            if let Some(last_step) = &mut last_step {
                field_path.0.push(ReferredField {
                    referring_id: last_step.referring_id,
                    referred_relation: last_step.referred_relation,
                    referred_id: last_step.referred_id,
                    referred_field: step.referring_id,
                    referred_field_name,
                });
                *last_step = Step {
                    referring_id: referred_field_name,
                    referred_relation: step.referred_relation,
                    referred_id: step.referred_id,
                };
            } else {
                last_step = Some(step);
            }
        }
        if let Some(last_step) = last_step {
            field_path.0.push(ReferredField {
                referring_id: last_step.referring_id,
                referred_relation: last_step.referred_relation,
                referred_id: last_step.referred_id,
                referred_field,
                referred_field_name,
            });
        }
        field_path
    }
}

impl<'a> Deref for FieldPath<'a> {
    type Target = Vec<ReferredField<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for FieldPath<'a> {
    type Item = ReferredField<'a>;
    type IntoIter = <Vec<ReferredField<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
