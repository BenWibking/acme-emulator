#ifndef __INCLUDE_HASH1D_H__
#define __INCLUDE_HASH1D_H__

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>

#include "read_hdf5.h"
#include "hash.h"

typedef struct MHash_ MHash;
struct MHash_
{
  double * bin_edges;
  size_t nbins;
  size_t * counts;
  size_t * allocated;
  halo_metadata ** h;
};

int compare_halo_metadata_by_id(const void* a, const void* b);
MHash* allocate_1d_hash(int nbins, double * bin_edges, size_t npoints, halo_metadata * h);
void sort_1d_hash(MHash * m);
void linearize_1d_hash(MHash * m, size_t len, halo_metadata * linear_halos);

#endif
