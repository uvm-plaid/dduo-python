# pyduet-dynamic

Differential privacy enables general statistical analysis of data with formal guarantees of privacy protection at the individual level. Tools that assist data analysts with utilizing differential privacy have frequently taken the form of programming languages and libraries. However, many existing programming languages designed for compositional verification of differential privacy impose significant burden on the programmer (in the form of complex type annotations). Supplementary library support for privacy analysis built on top of existing general-purpose languages has been more usable, but incapable of pervasive end-to-end enforcement of sensitivity analysis and privacy composition.

We introduce DDUO, a dynamic analysis for enforcing differential privacy. DDUO is usable by non-experts: its analysis is automatic and it requires no additional type annotations. DDUO can be implemented as a library for existing programming languages; we present a reference implementation in Python which features moderate runtime overheads on realistic workloads. We include support for several data types, distance metrics and operations which are commonly used in modern machine learning programs. We also provide initial support for tracking the sensitivity of data transformations in popular Python libraries for data analysis.

## Contributors
* Chike Abuah
* Alex Silence
* Vanessa White
* David Darais
* Joe Near

## Publications
https://arxiv.org/abs/2103.08805
