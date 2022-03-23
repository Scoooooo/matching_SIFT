IDIR =../include
CC=nvcc $(LINK) 
CFLAGS=-I$(IDIR)
LINK=  -lcublas -arch compute_86 


ODIR=obj
LDIR =../lib

LIBS=-lm

_DEPS = helper.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_DEPS = knn_brute.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_DEPS = lsh_h.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = test.o knn_brute.o lsh_h.o helper.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
.PHONY: clean, test 

test: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
