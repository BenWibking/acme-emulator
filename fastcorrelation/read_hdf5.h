#ifndef __INCLUDE_READ_HDF5_H__
#define __INCLUDE_READ_HDF5_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "hash.h"

typedef struct MHash_ MHash;

typedef struct
{
  float x;
  float y;
  float z;
} particle;

typedef struct
{
  float x;
  float y;
  float z;
  float mass;
  uint64_t id;
} halo;

typedef struct halo_metadata_ halo_metadata;
struct halo_metadata_
{
  uint64_t id;
  float mass;
  float density;
  float percentile;
};

void* read_particles_hdf5(char filename[], char dataset_name[], size_t *len);
void* read_halos_hdf5(char filename[], char dataset_name[], size_t *len);

herr_t write_halo_hdf5(char filename[], char dataset_name[], size_t len, halo_metadata* data);

#endif
