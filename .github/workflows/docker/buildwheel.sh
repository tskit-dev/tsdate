#!/bin/bash
DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DOCKER_DIR/shared.env"

set -e -x

ARCH=`uname -p`
echo "arch=$ARCH"

# We're running as root in the docker container so git commands issued by
# setuptools_scm will fail without this:
git config --global --add safe.directory /project
# Fetch the full history as we'll be missing tags otherwise.
git fetch --unshallow
for V in "${PYTHON_VERSIONS[@]}"; do
    PYBIN=/opt/python/$V/bin
    rm -rf build/       # Avoid lib build by narrow Python is used by wide python
    $PYBIN/python setup.py bdist_wheel
done

rm dist/*.tar.gz