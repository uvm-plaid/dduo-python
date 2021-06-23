# DDuo: dynamic analysis for differential privacy in Python

Differential privacy enables general statistical analysis of data with formal guarantees of privacy protection at the individual level. Tools that assist data analysts with utilizing differential privacy have frequently taken the form of programming languages and libraries. However, many existing programming languages designed for compositional verification of differential privacy impose significant burden on the programmer (in the form of complex type annotations). Supplementary library support for privacy analysis built on top of existing general-purpose languages has been more usable, but incapable of pervasive end-to-end enforcement of sensitivity analysis and privacy composition.

We introduce DDUO, a dynamic analysis for enforcing differential privacy. DDUO is usable by non-experts: its analysis is automatic and it requires no additional type annotations. DDUO can be implemented as a library for existing programming languages; we present a reference implementation in Python which features moderate runtime overheads on realistic workloads. We include support for several data types, distance metrics and operations which are commonly used in modern machine learning programs. We also provide initial support for tracking the sensitivity of data transformations in popular Python libraries for data analysis.

## Read the Paper

Details about DDuo can be found in the following paper:

*  C. Abuah, A. Silence, D. Darais and J. Near. "DDUO: General-Purpose
   Dynamic Analysis for Differential Privacy," in Proceedings of the
   34th IEEE Computer Security Foundations Symposium (CSF), 2021.
   
You can also find the paper [on arxiv](https://arxiv.org/abs/2103.08805).

## Try DDuo Now!

You can try DDuo in your browser! The following links will launch
example notebooks using the Binder service.

* Simple examples: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uvm-plaid/dduo-python/HEAD?filepath=notebooks%2FSimple%20Examples.ipynb)
* Additional examples: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uvm-plaid/dduo-python/HEAD?filepath=notebooks%2FAdditional%20Examples.ipynb)
* Gradient descent: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uvm-plaid/dduo-python/HEAD?filepath=notebooks%2FGradient%20Descent.ipynb)

## Install DDuo

To install DDuo locally:

1. Clone this repo
2. Navigate to the `dduo-python` directory
3. Type `pip install .` to install DDuo using `pip`

## Contributors
* [Chike Abuah](https://www.uvm.edu/~cabuah/)
* Alex Silence
* Vanessa White
* [David Darais](http://david.darais.com/)
* [Joe Near](https://www.uvm.edu/~jnear/)

