#ifndef __INCLUDE_HASH_H__
#define __INCLUDE_HASH_H__

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>

#include "read_hdf5.h"

#define M_PI 3.14159265358979323846

//#define TEST_ALL_PAIRS 1

#define FLOAT double
// remember: macro hygiene requires parenthesis around input quantities *and* output quantities
#define INDEX(i,j,k) ((k) + ((j) + ((i)*ngrid))*ngrid)
#define MAX(x,y) (((x)>(y)) ? (x) : (y))
#define SQ(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
// remember: must account for dx > 0.5*Lbox *and* dx < -0.5*Lbox!
#define PERIODIC(dx) (((dx)>0.5*Lbox) ? ((dx)-Lbox) : ( ((dx)<-0.5*Lbox) ? ((dx)+Lbox) : (dx) ))
//[GADGET macro] x=((x)>boxHalf_X)?((x)-boxSize_X):(((x)<-boxHalf_X)?((x)+boxSize_X):(x))

typedef struct {
  int x;
  int y;
  int z;
} grid_id;

typedef struct {
  double Lbox;
  int ngrid;
  int njack; /* the cube root of the number of jackknife samples */
  size_t * counts;
  size_t * allocated;
  FLOAT ** x;
  FLOAT ** y;
  FLOAT ** z;
  uint64_t ** id;
  FLOAT ** mass;
  grid_id ** sample_excluded_from;
} GHash;

void my_free(void* block);
void* my_malloc(size_t size);
void* my_realloc(void* old_block, size_t new_size, size_t old_size);

GHash* allocate_hash(int ngrid, int njack, double Lbox, size_t npoints, FLOAT * x, FLOAT * y, FLOAT * z);
void free_hash(GHash * g);
void geometric_hash(GHash * grid, FLOAT *x, FLOAT *y, FLOAT *z, size_t npoints);

void free_hash_with_id(GHash * g);
GHash* allocate_hash_with_id(int ngrid, double Lbox, size_t npoints, FLOAT * x, FLOAT * y, FLOAT * z, uint64_t * id);
void insert_particle_with_id(GHash * grid, FLOAT x, FLOAT y, FLOAT z, FLOAT mass, uint64_t id, size_t i);
void geometric_hash_with_id(GHash * grid, FLOAT *x, FLOAT *y, FLOAT *z, FLOAT *mass, \
			    uint64_t * id, size_t npoints);

void count_pairs_disjoint(FLOAT * restrict x, FLOAT * restrict y, FLOAT * restrict z, grid_id * restrict label, FLOAT * restrict adj_x, FLOAT * restrict adj_y, FLOAT * restrict adj_z, void * restrict adj_label, size_t count, size_t adj_count, uint64_t * pcounts, uint64_t * pcounts_jackknife, double *  bin_edges_sq, const int nbins, const int nsubsamples, const double Lbox);
void count_pairs_self(FLOAT * restrict x, FLOAT * restrict y, FLOAT * restrict z, grid_id * restrict label, size_t npoints, uint64_t * pcounts, uint64_t * pcounts_jackknife, double *  bin_edges_sq, const int nbins, const int nsubsamples, const double Lbox);

void count_pairs_naive(FLOAT * x, FLOAT * y, FLOAT * z, grid_id * label, size_t npoints, uint64_t * pcounts, uint64_t * pcounts_jackknife, double *  bin_edges_sq, const int nbins, const int nsubsamples, const double Lbox);
void count_pairs(GHash * restrict g, uint64_t * restrict pcounts, uint64_t * restrict pcounts_jackknife, double * restrict bin_edges_sq, int nbins);

void cross_count_pairs_naive(FLOAT * x1, FLOAT * y1, FLOAT * z1, grid_id * label1, size_t npoints1, FLOAT * x2, FLOAT * y2, FLOAT * z2, grid_id * label2, size_t npoints2, uint64_t * pcounts, uint64_t * pcounts_jackknife, double *  bin_edges_sq, const int nbins, const int njack, const double Lbox);
void cross_count_pairs(GHash * restrict g1, GHash * restrict g2, uint64_t * restrict pcounts, uint64_t * restrict pcounts_jackknife, double * restrict bin_edges_sq, int nbins, int njack);

void density_count_pairs(GHash * restrict g1, GHash * restrict g2, uint64_t * restrict halo_env_counts, uint64_t * restrict halo_id, FLOAT * halo_mass, double rmax_sq);
void _density_count_pairs(FLOAT x, FLOAT y, FLOAT z, FLOAT * restrict adj_x, FLOAT * restrict adj_y, FLOAT * restrict adj_z, size_t adj_count, uint64_t * pcounts, const double rmax_sq, const double Lbox);

#endif
