# Need to set PYTHONPATH so that we pick up the local tsdate
PYPATH=$(shell pwd)/../
TSDATE_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tsdate; print(tsdate.__version__.split("+")[0])')

BUILDDIR      = _build

all: dev

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tsdate version ${TSDATE_VERSION}
	sed -i -e s/__TSDATE_VERSION__/${TSDATE_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR $(BUILDDIR)