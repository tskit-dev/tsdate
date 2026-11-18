all: ext

ext:
	uv run python setup.py build_ext --inplace

clean:
	rm -f *.so *.o
	rm -fR build lib/build
