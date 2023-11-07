TREE_SEQS=\
        basic_example.trees \

all: ${TREE_SEQS}

%.trees:
	python make_examples.py

clean:
	rm -fR *.trees