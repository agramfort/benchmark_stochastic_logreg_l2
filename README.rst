Benchmark repository for Stochastic L2-regularized Logistic Regression
======================================================================

|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
The expectation of the L2-regularized Logistic loss consists in solving the
following program:

.. math::

    \min_w \mathbb{E}_{x,y} \left[ \log(1 + \exp(-y x^\top w)) + \frac{\lambda}{2} \|w\|_2^2 \right]

with:

.. math::

    y \in \{-1,1\}, x \in \mathbb{R}^{p}

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_stochastic_logreg_l2
   $ benchopt run ./benchmark_stochastic_logreg_l2

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_stochastic_logreg_l2 -s python_gd -d simulated --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_stochastic_logreg_l2/workflows/build/badge.svg
   :target: https://github.com/benchopt/benchmark_stochastic_logreg_l2/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
