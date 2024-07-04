//! # Some utilities to ease the implementation of Visitor / Acceptor in rust
//!
//! Visited structure do not have to be trees, they can be simple Directed Acyclic Graphs (DAGs).
//! The `accept` code will make sure the same node is not visited more than once.
//!
//! Use this code whenever possible.
//!

use std::{
    cmp::{Eq, PartialEq},
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    iter,
    ops::{Deref, DerefMut},
    vec::Vec,
};

/// The list of other acceptors an acceptor is depending on
pub struct Dependencies<'a, A>(Vec<&'a A>);

impl<'a, A> Dependencies<'a, A> {
    pub fn new(dependencies: Vec<&'a A>) -> Self {
        Dependencies(dependencies)
    }

    pub fn empty() -> Self {
        Dependencies(vec![])
    }
}

impl<'a, A> Deref for Dependencies<'a, A> {
    type Target = Vec<&'a A>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, A> DerefMut for Dependencies<'a, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, A> IntoIterator for Dependencies<'a, A> {
    type Item = <Vec<&'a A> as IntoIterator>::Item;
    type IntoIter = <Vec<&'a A> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, A, B: AsRef<[&'a A]>> From<B> for Dependencies<'a, A> {
    fn from(value: B) -> Self {
        value.as_ref().iter().cloned().collect()
    }
}

impl<'a, A> FromIterator<&'a A> for Dependencies<'a, A> {
    fn from_iter<T: IntoIterator<Item = &'a A>>(iter: T) -> Self {
        let dependencies: Vec<&'a A> = iter.into_iter().collect();
        Dependencies::new(dependencies)
    }
}

/// The list of other acceptors an acceptor is depending on, with their visited values
pub struct Visited<'a, A: PartialEq, O>(Vec<(&'a A, O)>);

impl<'a, A: PartialEq, O> Visited<'a, A, O> {
    pub fn new() -> Self {
        Visited(vec![])
    }

    fn push(&mut self, acceptor: &'a A, output: O) {
        self.0.push((acceptor, output))
    }

    pub fn get(&self, acceptor: &'a A) -> &O {
        &self.iter().find(|(a, _)| a == &acceptor).unwrap().1
    }

    pub fn pop(&mut self, acceptor: &'a A) -> O {
        let index = self.0.iter().position(|(a, _)| a == &acceptor).unwrap();
        self.0.swap_remove(index).1
    }

    pub fn find<P: Fn(&'a A) -> bool>(&self, predicate: P) -> Option<&O> {
        self.iter().find(|(a, _)| predicate(a)).map(|(_, o)| o)
    }
}

impl<'a, A: PartialEq, O> Deref for Visited<'a, A, O> {
    type Target = Vec<(&'a A, O)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, A: PartialEq, O> DerefMut for Visited<'a, A, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, A: PartialEq, O> IntoIterator for Visited<'a, A, O> {
    type Item = <Vec<(&'a A, O)> as IntoIterator>::Item;
    type IntoIter = <Vec<(&'a A, O)> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A generic visitor
pub trait Visitor<'a, A: Acceptor<'a>, O: Clone> {
    /// A function called on each node of a structured object with its dependencies already visited
    fn visit(&self, acceptor: &'a A, dependencies: Visited<'a, A, O>) -> O;
    /// Describe the dependencies of the acceptor
    /// The dependencies can be customized
    fn dependencies(&self, acceptor: &'a A) -> Dependencies<'a, A> {
        acceptor.dependencies()
    }
}
/// The identity Visitor
pub struct Identity;

impl<'a, A: Acceptor<'a>> Visitor<'a, A, &'a A> for Identity {
    fn visit(&self, acceptor: &'a A, _dependencies: Visited<'a, A, &'a A>) -> &'a A {
        acceptor
    }
}

pub type Iter<'a, A> =
    iter::FilterMap<Iterator<'a, &'a A, Identity, A>, fn((&'a A, State<&'a A>)) -> Option<&'a A>>;

pub type IterWith<'a, O, A, V> =
    iter::FilterMap<Iterator<'a, O, V, A>, fn((&'a A, State<O>)) -> Option<(&'a A, O)>>;

/// A generic acceptor trait
pub trait Acceptor<'a>: 'a + Sized + Debug + Eq + Hash {
    /// All the sub-objects to visit
    fn dependencies(&'a self) -> Dependencies<'a, Self>;

    fn accept<O: Clone, V: Visitor<'a, Self, O>>(&'a self, visitor: V) -> O {
        let mut state = State::Push;
        for (_a, s) in Iterator::new(visitor, self) {
            state = s
        }
        match state {
            State::Push => panic!("Found a `Push` state for Acceptor: {:?}. This should not be possible at this point.", self),
            State::Visit => panic!("Found a `Visit` state for Acceptor: {:?}. This should not be possible at this point.", self),
            State::Accept(output) => output.clone(),
        }
    }

    fn iter(&'a self) -> Iter<'a, Self> {
        Iterator::new(Identity, self).filter_map(|(_a, s)| match s {
            State::Accept(o) => Some(o),
            _ => None,
        })
    }

    fn iter_with<O: Clone, V: Visitor<'a, Self, O>>(
        &'a self,
        visitor: V,
    ) -> IterWith<'a, O, Self, V> {
        Iterator::new(visitor, self).filter_map(|(a, s)| match s {
            State::Accept(o) => Some((a, o)),
            _ => None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum State<O> {
    Push,
    Visit,
    Accept(O),
}

pub struct Iterator<'a, O: Clone, V: Visitor<'a, A, O>, A: Acceptor<'a>> {
    stack: Vec<&'a A>,
    state: HashMap<&'a A, State<O>>,
    visitor: V,
}

/// A visitor iterator implements roughly [DFS](https://en.wikipedia.org/wiki/Topological_sorting)
/// Acceptor, when submitted for visit are checked:
/// - if already accepted, nothing happens
/// - if already visited, fails because of cyclic graph
/// - if unknown, add as visited and visit its requirements.
impl<'a, O: Clone, V: Visitor<'a, A, O>, A: Acceptor<'a>> Iterator<'a, O, V, A> {
    pub fn new(visitor: V, acceptor: &'a A) -> Iterator<'a, O, V, A> {
        // Init the stack and state
        let stack = Vec::from([acceptor]);
        let state: HashMap<&'a A, State<O>> = HashMap::from([(acceptor, State::Push)]);
        Iterator {
            stack,
            state,
            visitor,
        }
    }
}

impl<'a, O: Clone, V: Visitor<'a, A, O>, A: Acceptor<'a>> iter::Iterator for Iterator<'a, O, V, A> {
    type Item = (&'a A, State<O>);

    fn next(&mut self) -> Option<(&'a A, State<O>)> {
        let acceptor = self.stack.pop()?;
        match self.state.get(&acceptor)? {
            // Get the status of the current acceptor
            State::Push => {
                // The acceptor node was just added
                self.state.insert(acceptor, State::Visit);
                self.stack.push(acceptor); // Push for visit
                for dependency in self.visitor.dependencies(acceptor).0 {
                    // Push its dependencies
                    match self.state.get(&dependency) {
                        Some(State::Push) => (),           // Process next
                        Some(State::Visit) => return None, // The node is already being taken care of
                        Some(State::Accept(_)) => (),      // The node has been processed
                        None => {
                            self.state.insert(dependency, State::Push);
                        }
                    }
                    self.stack.push(dependency);
                }
                Some((acceptor, State::Visit))
            }
            State::Visit => {
                // The acceptor node pushed its dependencies and is ready to run the visitor
                let mut dependencies = Visited::new();
                for dependency in self.visitor.dependencies(acceptor).0 {
                    // Push its dependencies
                    if let Some(State::Accept(o)) = self.state.get(&dependency) {
                        dependencies.push(dependency, o.clone());
                    } else {
                        return None;
                    }
                }
                let output = self.visitor.visit(acceptor, dependencies);
                self.state.insert(acceptor, State::Accept(output.clone()));
                Some((acceptor, State::Accept(output)))
            }
            State::Accept(_) => Some((acceptor, State::Push)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fmt, sync::Arc};

    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    struct Node(&'static str, Vec<Arc<Node>>);

    impl fmt::Display for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{} ({:?})", self.0, self as *const _)
        }
    }

    impl<'a> Acceptor<'a> for Node {
        fn dependencies(&'a self) -> Dependencies<'a, Self> {
            self.1.iter().map(|node| node.as_ref()).collect()
        }
    }

    struct DisplayVisitor;

    impl<'a> Visitor<'a, Node, String> for DisplayVisitor {
        fn visit(&self, acceptor: &'a Node, _dependencies: Visited<'a, Node, String>) -> String {
            format!("{acceptor}")
        }
    }

    fn build_diamond() -> Node {
        let a = Arc::new(Node("A", vec![]));
        let b = Arc::new(Node("B", vec![a.clone()]));
        let c = Arc::new(Node("C", vec![a]));
        Node("D", vec![b, c])
    }

    fn build_open_diamond() -> Node {
        let a = Node("A", vec![]);
        let b = Node("B", vec![Arc::new(a.clone())]);
        let c = Node("C", vec![Arc::new(a)]);
        Node("D", vec![Arc::new(b), Arc::new(c)])
    }

    #[test]
    fn test_diamond() {
        let open_diamond = build_open_diamond();
        for (_n, s) in open_diamond.iter_with(DisplayVisitor) {
            println!("{s}");
        }
        assert_eq!(open_diamond.iter_with(DisplayVisitor).count(), 4);
        println!("The labelling of processed nodes is based on their content, 2 copies of an object are considered the same");
        let diamond = build_diamond();
        for (_n, s) in diamond.iter_with(DisplayVisitor) {
            println!("{s}");
        }
        assert_eq!(diamond.iter_with(DisplayVisitor).count(), 4);
        println!("The labelling of processed nodes is based on their content, 2 copies of an object are considered the same");
    }

    fn build_semi_diamond() -> Node {
        let a = Arc::new(Node("A", vec![]));
        let b = Arc::new(Node("B", vec![a.clone()]));
        let c = Arc::new(Node("C", vec![a.clone(), b.clone()]));
        (*c).clone()
    }

    #[test]
    fn test_semi_diamond() {
        let semi_diamond = build_semi_diamond();
        for (_n, s) in semi_diamond.iter_with(DisplayVisitor) {
            println!("{s}");
        }
        // assert_eq!(double_diamond.iter_with(DisplayVisitor).count(), 4);
        println!("The half diamond")
    }
}
