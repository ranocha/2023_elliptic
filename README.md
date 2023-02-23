# A discontinuous Galerkin discretization of elliptic problems with improved convergence properties using summation by parts operators

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{ranocha2023discontinuous,
  title={A discontinuous {G}alerkin discretization of elliptic problems with
         improved convergence properties using summation by parts operators},
  author={Ranocha, Hendrik},
  year={2023},
  month={TODO},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={TODO}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{ranocha2023discontinuousRepro,
  title={Reproducibility repository for
         "{A} discontinuous {G}alerkin discretization of elliptic problems with
         improved convergence properties using summation by parts operators"},
  author={Ranocha, Hendrik},
  year={2023},
  howpublished={\url{TODO}},
  doi={TODO}
}
```

## Abstract

Nishikawa (2007) proposed to reformulate the classical Poisson equation as a
steady state problem for a linear hyperbolic system. This results in optimal
error estimates for both the solution of the elliptic equation and its gradient.
However, it prevents the application of well-known solvers for elliptic
problems. We show connections to a discontinuous Galerkin (DG) method analyzed
by Cockburn, Guzmán, and Wang (2009) that is very difficult to implement in
general. Next, we demonstrate how this method can be implemented efficiently
using summation by parts (SBP) operators, in particular in the context of
SBP DG methods such as DG spectral element method (DGSEM). The resulting scheme
combines nice properties of both the hyperbolic and the elliptic point of view,
in particular a high order of convergence of the gradients, which is one order
higher than what one would usually expect from DG methods for elliptic problems.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.8.3.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface.

Then, you need to start Julia in the `code` directory and follow the
instructions in the `README.md` file therein.


## Authors

- [Hendrik Ranocha](https://ranocha.de) (University of Hamburg, Germany)


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
