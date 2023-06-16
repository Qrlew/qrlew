# [Qrlew](https://qrlew.github.io/) framework (by [Sarus](https://www.sarus.tech/))
Open source SQL manipulation framework written in Rust

## What is [Qrlew](https://qrlew.github.io/)?
[Qrlew](https://qrlew.github.io/) is an open source library that aims to parse and compile SQL queries into an Intermediate Representation (IR) that is well-suited for various rewriting tasks. Although it was originally designed for privacy-focused applications, it can be utilized for a wide range of purposes.

### SQL Query IR
[Qrlew](https://qrlew.github.io/) transforms a SQL query into a combination of simple operations such as Map, Reduce and Join that are applied to Tables. This representation simplifies the process of rewriting queries and reduces dependencies on the diverse range of syntactic constructs present in SQL.

### Type Inference Engine
Differential Privacy (DP) guaranrtees are hard to obtain without destroying too much information. In many mechanisms having prior bounds on values can improve the utility of DP results dramatically. By propagating types cleverly, [Qrlew](https://qrlew.github.io/) can returns bounds for all values.

### Differential Privacy compiler
[Qrlew](https://qrlew.github.io/) can compile SQL queries into Differentially Private ones. The process is inspired by Wilson et al. 2020. The complexity of the compilation process makes [Qrlew](https://qrlew.github.io/) IR very useful at delivering clean, readable and reliable code.
