#include "read_hdf5.h"

herr_t write_gal_hdf5(char filename[], char dataset_name[], size_t len, HODgal* data) {
  /* open HDF5 file*/
  hid_t memtype,filetype;
  hid_t file_id, dataset, space;
  hsize_t dims[1] = {len};
  herr_t status;

  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  memtype = H5Tcreate(H5T_COMPOUND, sizeof(HODgal));
  status = H5Tinsert(memtype, "x", HOFFSET(HODgal,X), H5T_NATIVE_FLOAT);
  status = H5Tinsert(memtype, "y", HOFFSET(HODgal,Y), H5T_NATIVE_FLOAT);
  status = H5Tinsert(memtype, "z", HOFFSET(HODgal,Z), H5T_NATIVE_FLOAT);
  status = H5Tinsert(memtype, "weight", HOFFSET(HODgal,weight), H5T_NATIVE_FLOAT);
  status = H5Tinsert(memtype, "is_sat", HOFFSET(HODgal,is_sat), H5T_NATIVE_INT);
  status = H5Tinsert(memtype, "halo_mass", HOFFSET(HODgal,halo_mass), H5T_NATIVE_FLOAT);

  size_t float_size_on_disk = H5Tget_size(H5T_IEEE_F32LE); // single precision float
  size_t int_size_on_disk = H5Tget_size(H5T_STD_I32LE); // int32_t
  size_t offset = 0;
  filetype = H5Tcreate(H5T_COMPOUND, 4*float_size_on_disk + int_size_on_disk + float_size_on_disk);
  status = H5Tinsert(filetype, "x", offset, H5T_IEEE_F32LE);
  offset += float_size_on_disk;
  status = H5Tinsert(filetype, "y", offset, H5T_IEEE_F32LE);
  offset += float_size_on_disk;
  status = H5Tinsert(filetype, "z", offset, H5T_IEEE_F32LE);
  offset += float_size_on_disk;
  status = H5Tinsert(filetype, "weight", offset, H5T_IEEE_F32LE);
  offset += float_size_on_disk;
  status = H5Tinsert(filetype, "is_sat", offset, H5T_STD_I32LE);
  offset += int_size_on_disk;
  status = H5Tinsert(filetype, "halo_mass", offset, H5T_IEEE_F32LE);
  offset += float_size_on_disk;

  space = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(file_id, dataset_name, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset,memtype,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

  H5Fclose(file_id);
  return status;
}
