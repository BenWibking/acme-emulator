HDF5_C_INCLUDE=-I$(HOME)/anaconda3/envs/acme/include
HDF5_C_LIBS=-L$(HOME)/anaconda3/envs/acme/lib -lm -lhdf5 -lhdf5_hl

CC=gcc
CFLAGS=-Wall -O3 -std=c99

#CC=icc
#CFLAGS="-Wall -O3 -ipo -qopenmp -std=c99"

INCLDIRS = $(HDF5_C_INCLUDE) -I ../
LFLAGS = $(HDF5_C_LIBS) -lgsl -lgslcblas
SOURCES = read_hdf5.c write_hdf5.c NFW_CDF.c hod.c compute_mocks.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = read_hdf5.h
EXEC = compute_mocks

.c.o:
	$(CC) $(CFLAGS) $(INCLDIRS) -c $<

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJECTS) $(LFLAGS)

$(OBJECTS): $(HEADERS) Makefile

clean:
	rm -f $(EXEC) *~ $(OBJECTS)


