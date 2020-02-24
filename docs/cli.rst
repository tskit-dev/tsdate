.. _sec_cli:

======================
Command line interface
======================

``tsdate`` provides a Command Line Interface as a simple interface with the :ref:`Python API <sec_python_api>`.


.. code-block:: bash

    $ tsdate

or

.. code-block:: bash

    $ python3 -m tsdate

The second means of using the CLI is useful when multiple versions of Python are 
installed or if the :command:`tsdate` executable is not installed on your path.

++++++++++++++++
Argument details
++++++++++++++++

.. argparse::
    :module: tsdate.cli
    :func: tsdate_cli_parser
    :prog: tsdate
    :nodefault:
