CC=gcc
CFLAGS=-g -O3 -Wall -march=native -std=c99
#CC=icc
#CFLAGS=-g -O3 -Wall -march=native -std=c99

HDF5_C_INCLUDE=-I$(CONDA_PREFIX)/include
HDF5_C_LIBS=-Wl,-rpath $(CONDA_PREFIX)/lib -L$(CONDA_PREFIX)/lib -lhdf5

INCLUDE=`gsl-config --cflags` $(HDF5_C_INCLUDE) $(MPI_CFLAGS)
LIB=`gsl-config --libs` $(HDF5_C_LIBS)

OBJS_AUTO=hash.o auto_counts.o read_hdf5.o main.o
OBJS_TEST_AUTO=hash.o auto_counts.o read_hdf5.o test_auto.o
OBJS_CROSS=hash.o auto_counts.o cross_counts.o read_hdf5.o main_cross.o
OBJS_TEST_CROSS=hash.o auto_counts.o cross_counts.o read_hdf5.o test_cross.o
OBJS_DENSITY=hash.o hash1d.o density_counts.o read_hdf5.o write_hdf5.o main_density.o
EXEC_AUTO = auto
EXEC_CROSS = cross
EXEC_DENSITY = density
EXEC_TEST_AUTO = test_auto
EXEC_TEST_CROSS = test_cross

default: auto cross density test_auto test_cross

density-test: $(EXEC_DENSITY)
	./$(EXEC_DENSITY) 30 720 galaxy_mock.hdf5 DM_subsample.hdf5

test: $(EXEC_AUTO)
	./$(EXEC_AUTO) --test-pairs 30 0.1 30 720 1 galaxy_mock.hdf5 
	./$(EXEC_AUTO) --test-pairs 30 0.1 30 720 1 DM_subsample.hdf5

clean:
	rm *.o; rm $(EXEC_AUTO) $(EXEC_CROSS) $(EXEC_DENSITY) $(EXEC_TEST_AUTO) $(EXEC_TEST_CROSS)

main.o: main.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

main_cross.o: main_cross.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

main_density.o: main_density.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

read_hdf5.o: read_hdf5.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

write_hdf5.o: write_hdf5.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

auto_counts.o: auto_counts.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

cross_counts.o: cross_counts.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

density_counts.o: density_counts.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

hash.o: hash.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

hash1d.o: hash1d.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

test_auto.o: test_auto.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

test_cross.o: test_cross.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

test_auto: $(OBJS_TEST_AUTO)
	$(CC) $(CFLAGS) $(OBJS_TEST_AUTO) $(LIB) -o $(EXEC_TEST_AUTO)

test_cross: $(OBJS_TEST_CROSS)
	$(CC) $(CFLAGS) $(OBJS_TEST_CROSS) $(LIB) -o $(EXEC_TEST_CROSS)

auto: $(OBJS_AUTO)
	$(CC) $(CFLAGS) $(OBJS_AUTO) $(LIB) -o $(EXEC_AUTO)

cross: $(OBJS_CROSS)
	$(CC) $(CFLAGS) $(OBJS_CROSS) $(LIB) -o $(EXEC_CROSS)

density: $(OBJS_DENSITY)
	$(CC) $(CFLAGS) $(OBJS_DENSITY) $(LIB) -o $(EXEC_DENSITY)
