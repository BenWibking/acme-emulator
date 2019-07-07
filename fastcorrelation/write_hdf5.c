#include "read_hdf5.h"

herr_t write_halo_hdf5(char filename[], char dataset_name[], size_t len, halo_metadata* data) {
  /* open HDF5 file*/
  hid_t memtype,filetype;
  hid_t file_id, dataset, space;
  hsize_t dims[1] = {len};
  herr_t status;

  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  memtype = H5Tcreate(H5T_COMPOUND, sizeof(halo));
  //  status = H5Tinsert(memtype, "id", HOFFSET(halo_metadata,id), H5T_NATIVE_ULONG);
  status = H5Tinsert(memtype, "mass", HOFFSET(halo_metadata,mass), H5T_NATIVE_FLOAT);
  status = H5Tinsert(memtype, "density", HOFFSET(halo_metadata,density), H5T_NATIVE_FLOAT);
  status = H5Tinsert(memtype, "percentile", HOFFSET(halo_metadata,percentile), H5T_NATIVE_FLOAT);

  size_t float_size_on_disk = H5Tget_size(H5T_IEEE_F32BE); // single precision
  size_t uint64_size_on_disk = H5Tget_size(H5T_STD_U64LE); // 'LE' = little endian
  size_t offset = 0;
  //  filetype = H5Tcreate(H5T_COMPOUND, uint64_size_on_disk + 3*float_size_on_disk);
  filetype = H5Tcreate(H5T_COMPOUND, 3*float_size_on_disk);
  //  status = H5Tinsert(filetype, "id", offset, H5T_STD_U64LE);
  //  offset += uint64_size_on_disk;
  status = H5Tinsert(filetype, "mass", offset, H5T_IEEE_F32BE);
  offset += float_size_on_disk;
  status = H5Tinsert(filetype, "density", offset, H5T_IEEE_F32BE);
  offset += float_size_on_disk;
  status = H5Tinsert(filetype, "percentile", offset, H5T_IEEE_F32BE);
  offset += float_size_on_disk;

  space = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(file_id, dataset_name, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset,memtype,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

  H5Fclose(file_id);
  return status;
}


