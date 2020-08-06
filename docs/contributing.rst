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
2.  `Software, Methods, and Data for Statistical Genetics forum`_: for general discussion

Discussions on the forum tend to be about higher-level themes, and statistical genetics in
general. Coding details should be discussed on GitHub issues and pull requests.

.. _`GitHub Issue Tracker`: https://github.com/pystatgen/sgkit/issues
.. _`Software, Methods, and Data for Statistical Genetics forum`: https://discourse.smadstatgen.org/


Code repositories
-----------------

Code and documentation for *sgkit* is maintained in a few git repositories hosted on the
GitHub ``pystatgen`` organization, https://github.com/pystatgen.  This includes the primary
repository and several other repositories for different components.  A
non-exhaustive list follows:

*  https://github.com/pystatgen/sgkit: The main code repository containing the
   data representations (in Xarray), algorithms, and most documentation
*  https://github.com/pystatgen/sgkit-bgen: Code for reading BGEN format files
   into *sgkit*'s Xarray represention
*  https://github.com/pystatgen/sgkit-plink: Code for reading PLINK format files
   into *sgkit*'s Xarray represention

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
you are working on it.

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

To build the library you can install the necessary requirements using
pip::

  cd sgkit
  pip install -r requirements.txt -r requirements-dev.txt

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

Python versions
~~~~~~~~~~~~~~~

*sgkit* supports Python versions 3.7 and 3.8.


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


Docstrings
~~~~~~~~~~

User facing functions should follow the numpydoc_ standard, including
sections for ``Parameters``, ``Examples``, and general explanatory prose.

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

You can now make edits to rst files and run ``make html`` again to update
the affected pages.

The documentation build is checked by CI to ensure that it builds
without warnings. You can do that locally with::

   make clean html SPHINXOPTS="-W --keep-going"

.. _Sphinx: https://www.sphinx-doc.org/


Review process
--------------

Pull requests will be reviewed by a project maintainer. All changes to *sgkit* require
approval by at least one maintainer.
