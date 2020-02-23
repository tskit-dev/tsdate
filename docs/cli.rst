.. _sec_cli:

======================
Command line interface
======================

``tsdate`` provides a Command Line Interface as a simple interface with the :ref:`Python API <sec_api>`.


.. code-block:: bash

    $ tsdate

or

.. code-block:: bash

    $ python3 -m tsdate

The first form is more intuitive and works well most of the time. The second
form is useful when multiple versions of Python are installed or if the
:command:`tsdate` executable is not installed on your path.

++++++++++++++++
Argument details
++++++++++++++++

.. argparse::
    :module: tsdate
    :func: get_cli_parser
    :prog: tsdate
    :nodefault:
