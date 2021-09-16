IDIR =../include
CC=nvcc $(LINK) 
CFLAGS=-I$(IDIR)

LINK=-lcublas -lcurand


ODIR=obj
LDIR =../lib

LIBS=-lm

_DEPS = knn_brute.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_DEPS = lsh.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = test.o knn_brute.o lsh.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
.PHONY: clean, test 

test: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
