#include "read_hdf5.h"

void* read_particles_hdf5(char filename[], char dataset_name[], size_t *len) {
  /* open HDF5 file*/
  /* in: filename, dataset_name
     out: len */

  hid_t halo_tid;
  hid_t file_id, dataset, space;
  hsize_t dims[2];

  file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen(file_id, dataset_name, H5P_DEFAULT);

  space = H5Dget_space(dataset);
  H5Sget_simple_extent_dims(space, dims, NULL);
  particle *data = (particle*) malloc(dims[0]*sizeof(particle));

  halo_tid = H5Tcreate(H5T_COMPOUND, sizeof(particle));
  H5Tinsert(halo_tid, "x", HOFFSET(particle,x), H5T_NATIVE_FLOAT);
  H5Tinsert(halo_tid, "y", HOFFSET(particle,y), H5T_NATIVE_FLOAT);
  H5Tinsert(halo_tid, "z", HOFFSET(particle,z), H5T_NATIVE_FLOAT);

  H5Dread(dataset, halo_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  *len = dims[0];

  H5Fclose(file_id);

  return (void*)data;
}

void* read_halos_hdf5(char filename[], char dataset_name[], size_t *len) {
  /* open HDF5 file*/
  /* in: filename, dataset_name
     out: len */

  hid_t halo_tid;
  hid_t file_id, dataset, space;
  hsize_t dims[2];

  file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen(file_id, dataset_name, H5P_DEFAULT);

  space = H5Dget_space(dataset);
  H5Sget_simple_extent_dims(space, dims, NULL);
  halo *data = (halo*) malloc(dims[0]*sizeof(halo));

  halo_tid = H5Tcreate(H5T_COMPOUND, sizeof(halo));
  H5Tinsert(halo_tid, "x", HOFFSET(halo,x), H5T_NATIVE_FLOAT);
  H5Tinsert(halo_tid, "y", HOFFSET(halo,y), H5T_NATIVE_FLOAT);
  H5Tinsert(halo_tid, "z", HOFFSET(halo,z), H5T_NATIVE_FLOAT);
  H5Tinsert(halo_tid, "mass", HOFFSET(halo,mass), H5T_NATIVE_FLOAT);
  H5Tinsert(halo_tid, "id", HOFFSET(halo,id), H5T_NATIVE_ULONG);

  H5Dread(dataset, halo_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  *len = dims[0];

  H5Fclose(file_id);

  return (void*)data;
}
