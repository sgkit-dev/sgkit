.. _contributing:

**********************
Contributing to sgkit
**********************

.. contents:: Table of contents:
   :local:

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.
This page provides resources on how best to contribute.

.. note::

  Large parts of this document came from the `Dask Development Guidelines
  <https://docs.dask.org/en/latest/develop.html>`_.


Discussion forums
-----------------

Conversation about *sgkit* happens in the following places:

1.  `GitHub Issue Tracker`_: for discussions around new features or bugs
2.  `GitHub Discussions`_: for general discussions, and questions like "how do I do X",
    or "what's the best way to do Y"?
3.  `Python for Statistical Genetics forum`_: for general discussion (deprecated)


Discussions on GitHub Discussions (and previously the forum) tend to be about higher-level themes, and statistical genetics in
general. Coding details should be discussed on GitHub issues and pull requests.

.. _`GitHub Issue Tracker`: https://github.com/pystatgen/sgkit/issues
.. _`GitHub Discussions`: https://github.com/pystatgen/sgkit/discussions
.. _`Python for Statistical Genetics forum`: https://discourse.pystatgen.org/


Code repositories
-----------------

Code and documentation for *sgkit* is maintained in a few git repositories hosted on the
GitHub ``pystatgen`` organization, https://github.com/pystatgen.  This includes the primary
repository and several other repositories for different components.  A
non-exhaustive list follows:

*  https://github.com/pystatgen/sgkit: The main code repository containing the
   data representations (in Xarray), algorithms, and most documentation

Git and GitHub can be challenging at first.  Fortunately good materials exist
on the internet.  Rather than repeat these materials here, we refer you to
Pandas' documentation and links on this subject at
https://pandas.pydata.org/pandas-docs/stable/contributing.html


Issues
------

The community discusses and tracks known bugs and potential features in the
`GitHub Issue Tracker`_.  If you have a new idea or have identified a bug, then
you should raise it there to start public discussion.

If you are looking for an introductory issue to get started with development,
then check out the `"good first issue" label`_, which contains issues that are good
for starting developers.  Generally, familiarity with Python, NumPy, and
some parallel computing (Dask) are assumed.

.. _`"good first issue" label`: https://github.com/pystatgen/sgkit/labels/good%20first%20issue

Before starting work, make sure there is an issue covering the feature or bug you
plan to produce a pull request for. Assign the issue to yourself to indicate that
you are working on it. In the PR make sure to mention/link the related issue(s).

Development environment
-----------------------

Download code
~~~~~~~~~~~~~

Make a fork of the main `sgkit repository <https://github.com/pystatgen/sgkit>`_ and
clone the fork::

   git clone https://github.com/<your-github-username>/sgkit

Contributions to *sgkit* can then be made by submitting pull requests on GitHub.


Install
~~~~~~~

To build the library you need to first install GSL (GNU Scientific Library),
which is required for ``msprime``. Since it's a system package and installation
may vary depending on the operating system, the instructions for the same can be
found on `msprime docs. <https://msprime.readthedocs.io/en/stable/installation.html#installing-system-requirements>`_

You can install rest of the necessary requirements using pip::

  cd sgkit
  pip install -r requirements.txt -r requirements-dev.txt -r requirements-doc.txt


Also install pre-commit, which is used to enforce coding standards::

   pre-commit install


Run tests
~~~~~~~~~

*sgkit* uses pytest_ for testing.  You can run tests from the main ``sgkit`` directory
as follows::

   pytest

.. _pytest: https://docs.pytest.org/en/latest/


Contributing to code
--------------------

*sgkit* maintains development standards that are similar to most PyData projects.  These standards include
language support, testing, documentation, and style.

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

*sgkit* uses GitHub Actions as a Continuous Integration (CI) service to check code
contributions. Every push to every pull request on GitHub will run the tests,
check test coverage, check coding standards, and check the documentation build.


Test
~~~~

*sgkit* employs extensive unit tests to ensure correctness of code both for today
and for the future.

Test coverage must be 100% for code to be accepted. You can measure the coverage
on your local machine by running::

   pytest --cov=sgkit --cov-report=html

A report will be written in the ``htmlcov`` directory that will show any lines that
are not covered by tests.

The test suite is run automatically by CI.

Test files live in ``sgkit/tests`` directory, test filename naming convention:
``test_<MODULE>.py``.

Use double underscore to organize tests into groups, for example:

.. code-block:: python

   def test_foo__accepts_empty_input():
     ...

   def test_foo__accepts_strings():
     ...


Docstrings
~~~~~~~~~~

User facing functions should follow the numpydoc_ standard, including
sections for ``Parameters``, ``Examples``, and general explanatory prose.

The types for parameters and returns should not be added to the docstring,
they should be only added as type hints, to avoid duplication.

A reference for each new public function should be added in the API documentation file
``docs/api.rst``, which makes them accessible on the user documentation page.

By default, examples will be doc-tested.  Reproducible examples in documentation
is valuable both for testing and, more importantly, for communication of common
usage to the user.  Documentation trumps testing in this case and clear
examples should take precedence over using the docstring as testing space.
To skip a test in the examples add the comment ``# doctest: +SKIP`` directly
after the line.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Docstrings are tested by CI. You can test them locally
by running ``pytest`` (this works because the ``--doctest-modules`` option is automatically added
in the *setup.cfg* file).


Coding standards
~~~~~~~~~~~~~~~~

*sgkit* uses `pre-commit <https://pre-commit.com/>`_ to enforce coding standards. Pre-commit
runs when you commit code to your local git repository, and the commit will only succeed
if the change passes all the checks. It is also run for pull requests using CI.

*sgkit* uses the following tools to enforce coding standards:

1.  `Black <https://black.readthedocs.io/en/stable/>`_: for code formatting
2.  `Flake8 <http://flake8.pycqa.org/en/latest/>`_: for style consistency
3.  `isort <https://timothycrosley.github.io/isort/>`_: for import ordering
4.  `mypy <http://mypy-lang.org/>`_: for static type checking

To manually enforce (or check) the source code adheres to our coding standards without
doing a git commit, run::

   pre-commit run --all-files

To run a specific tool (``black``/``flake8``/``isort``/``mypy`` etc)::

   pre-commit run black --all-files

You can omit ``--all-files`` to only check changed files.


PR/Git ops
~~~~~~~~~~

We currently use ``rebase`` or ``squash`` PR merge strategies. This means that
following certain git best practices will make your development life easier.


1. Try to create isolated/single issue PRs

   This makes it easier to review your changes, and should guarantee
   a speedy review.

2. Try to push meaningful small commits

   Again this makes it easier to review your code, and in case of
   bugs easier to isolate specific buggy commits.


Please read `git best practices <https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_public_project>`_
and specifically a very handy `interactive rebase doc <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_rewriting_history>`_.


Contributing to documentation
-----------------------------

*sgkit* uses Sphinx_ for documentation, hosted at https://pystatgen.github.io/sgkit/.
Documentation is maintained in the RestructuredText markup language (``.rst``
files) in ``docs``.  The documentation consists both of prose
and API documentation.

You can build the documentation locally with ``make``::

   cd docs
   make html

The resulting HTML files end up in the ``_build/html`` directory.

You can now make edits to ``.rst`` files and run ``make html`` again to update
the affected pages.

The documentation build is checked by CI to ensure that it builds
without warnings. You can do that locally with::

   make clean html SPHINXOPTS="-W --keep-going -n"

.. _Sphinx: https://www.sphinx-doc.org/

Benchmarks
----------

*sgkit* uses asv_ (Airspeed Velocity) for micro benchmarking.
Airspeed Velocity manages building the environment via conda itself. The
recipe for the same is defined in the ``benchmarks/asv.conf.json``
configuration file. The benchmarks should be written in the ``benchmarks/``
directory. For more information on different types of benchmarks have a look
at the ``asv`` documentation here: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#writing-benchmarks

The results of benchmarks are uploaded to benchmarks repository: https://github.com/pystatgen/sgkit-benchmarks-asv
via Github Actions. They can be seen on the static site here: https://pystatgen.github.io/sgkit-benchmarks-asv

You can run the benchmark suite locally with::

   asv run --config benchmarks/asv.conf.json

You can generate the html of the results via::

  asv publish --config benchmarks/asv.conf.json -v

The resulting HTML files end up in the ``benchmarks/html`` directory and the
results in ``benchmarks/results`` directory.

You can see the results of the benchmarks in the browser by running a local
server::

  asv preview --config benchmarks/asv.conf.json -v

.. _asv: https://www.sphinx-doc.org/

The benchmark machine is the Github Actions machine, which has roughly the
following configurations::

  {
      "arch": "x86_64",
      "cpu": "Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz",
      "machine": "fv-az183-669",
      "num_cpu": "2",
      "os": "Linux 5.4.0-1039-azure",
      "ram": "7121276",
      "version": 1
  }

The above configuration was determined by running the following command on
Github Actions, on one of the runs::

  asv machine --yes

The configuration above does changes slightly in every run, for example we
could get a machine with different cpu like say the one with 2.30GHz or the one
with slightly less RAM (not a huge deviation from above though). As of now it
is not possible to fix this, unless we use a custom machine for benchmarking,
hence minor deviation in benchmarks performance should be consumed with a pinch
of salt.

Review process
--------------

Pull requests will be reviewed by a project maintainer. All changes to *sgkit* require
approval by at least one maintainer.

We use `mergify <https://mergify.io/>`_ to automate PR flow. A project
`committer <https://github.com/orgs/pystatgen/teams/committers>`_ (reviewer) can decide
to automatically merge a PR by labeling it with ``auto-merge``, and then when the PR gets
at least one approval from a committer and a clean build it will get merged automatically.

Design discussions
------------------

The information on these topics may be useful for developers in understanding the
history behind the design choices that have been made within the project so far.

Dataset subclassing
~~~~~~~~~~~~~~~~~~~

Debates on whether or not we should use Xarray objects directly or
put them behind a layer of encapsulation:

- https://github.com/pystatgen/sgkit/pull/16#issuecomment-657725092
- https://github.com/pystatgen/sgkit/pull/78#issuecomment-669878845

Dataset API typing
~~~~~~~~~~~~~~~~~~

Discussions around bringing stricter array type enforcement into the API:

- https://github.com/pystatgen/sgkit/issues/43
- https://github.com/pystatgen/sgkit/pull/124
- https://github.com/pystatgen/sgkit/pull/276


Delayed invariant checks
~~~~~~~~~~~~~~~~~~~~~~~~

Discussions on how to run sanity checks on arrays efficiently and why those checks would be
useful if they were possible (they are not possible currently w/ Dask):

- https://github.com/pystatgen/sgkit/issues/61
- https://github.com/dask/dask/issues/97

Mixed ploidy
~~~~~~~~~~~~

Proposal for handling mixed ploidy: https://github.com/pystatgen/sgkit/issues/243

Numba guvectorize usage
~~~~~~~~~~~~~~~~~~~~~~~

Learning how to use ``guvectorize`` effectively:

- https://github.com/pystatgen/sgkit/pull/114
- https://github.com/pystatgen/sgkit/pull/348

API namespace
~~~~~~~~~~~~~

Sgkit controls API namespace via init files. To accommodate for mypy and docstrings
we include both imports and ``__all__`` declaration. More on this decision in the issue:
https://github.com/pystatgen/sgkit/issues/251
