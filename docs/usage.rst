.. usage:

*****
Usage
*****

.. contents:: Table of contents:
   :local:


.. _dataset_merge:

Dataset merge behavior
======================

Generally, method functions in sgkit compute some new variables based on the
input dataset, then return a new output dataset that consists of the input
dataset plus the new computed variables. The input dataset is unchanged.

This behavior can be controlled using the ``merge`` parameter. If set to ``True``
(the default), then the function will merge the input dataset and the computed
output variables into a single dataset. Output variables will overwrite any
input variables with the same name, and a warning will be issued in this case.
If ``False``, the function will return only the computed output variables.