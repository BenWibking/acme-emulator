#include "hash.h"
#include "hash1d.h"

//#define PRINT_HALOS

int main(int argc, char *argv[])
{
  /* input: number of points to test, number of cells per axis */
  double Lbox, rmax, logdM;
  float input_boxsize, input_rmax, input_logdm;
  char *filenameA, *filenameB, *output_filename;
  int njack = 1;

  /* check inputs */
  if(argc != 7) {
    printf("./density rmax box_size logdM halos_filenameA particles_filenameB output_filename\n");
    exit(-1);
  }

  /* default rmax should be 10 h^-1 Mpc */

  input_rmax = atof(argv[1]);
  input_boxsize = atof(argv[2]);
  input_logdm = atof(argv[3]);

  filenameA = malloc(sizeof(char)*(strlen(argv[4])+1));
  sprintf(filenameA,"%s",argv[4]);
  filenameB = malloc(sizeof(char)*(strlen(argv[5])+1));
  sprintf(filenameB,"%s",argv[5]);
  output_filename = malloc(sizeof(char)*(strlen(argv[6])+1));
  sprintf(output_filename,"%s",argv[6]);

  if(input_boxsize <= 0.) {
    printf("boxsize must be positive!\n");
    exit(1);
  }
  if(input_logdm <= 0.) {
    printf("logdM must be positive!\n");
    exit(1);
  }

  size_t npointsA,npointsB;
  int ngrid;
  rmax = input_rmax;
  Lbox = (double)input_boxsize;
  logdM = input_logdm;
  /* compute ngrid from rmax */
  ngrid = (int)floor(Lbox/rmax);

  /* read from file */
  halo *pointsA = read_halos_hdf5(filenameA, "halos", &npointsA); // halos
  particle *pointsB = read_particles_hdf5(filenameB, "particles", &npointsB); // particles

  // separate arrays (or Fortran-style arrays) are necessary both for SIMD and cache efficiency
  FLOAT *x1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  FLOAT *y1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  FLOAT *z1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  uint64_t *halo_id = (uint64_t*) my_malloc(npointsA*sizeof(uint64_t));
  FLOAT *halo_mass = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));

  FLOAT *x2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  FLOAT *y2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  FLOAT *z2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  
  size_t n;
  for(n=0;n<npointsA;n++)
    {
      x1[n] = pointsA[n].x;
      y1[n] = pointsA[n].y;
      z1[n] = pointsA[n].z;
      halo_id[n] = n; // use this to re-order output to be the same as the input
      halo_mass[n] = pointsA[n].mass;
    }
  for(n=0;n<npointsB;n++)
    {
      x2[n] = pointsB[n].x;
      y2[n] = pointsB[n].y;
      z2[n] = pointsB[n].z;
    }

  free(pointsA);
  free(pointsB);

  /* hash into grid cells */
  GHash *grid1 = allocate_hash_with_id(ngrid, Lbox, npointsA, x1, y1, z1, halo_id);
  GHash *grid2 = allocate_hash(ngrid, njack, Lbox, npointsB, x2, y2, z2);
  if ((int)grid1 == 0) {
    printf("allocating grid1 failed!\n");
    exit(-1);
  }
  if ((int)grid2 == 0) {
    printf("allocating grid2 failed!\n");
    exit(-1);
  }

  geometric_hash_with_id(grid1, x1, y1, z1, halo_mass, halo_id, npointsA);
  geometric_hash(grid2, x2, y2, z2, npointsB);  

  /* free 1d arrays */
  my_free(x1);
  my_free(y1);
  my_free(z1);
  my_free(x2);
  my_free(y2);
  my_free(z2);


  /* compute pair counts */
  uint64_t *halo_env_counts = my_malloc(npointsA*sizeof(uint64_t));
  int i;
  for(i=0;i<npointsA;i++) {
    halo_env_counts[i] = (long int) 0;
    halo_id[i] = (long int) 0; // re-use halo_id array
    halo_mass[i] = 0.; // re-use halo_mass array
  }

  /* count particles within rmax of each halo */
  density_count_pairs(grid1, grid2, halo_env_counts, halo_id, halo_mass, rmax*rmax);

  free_hash_with_id(grid1);
  free_hash(grid2);

  /* compute mass bins */
  double binmin = 10.0; // log10 Msun
  double binmax = 15.0; // log10 Msun
  double deltabin = logdM;
  int nbins = ((int) floor((binmax - binmin)/deltabin)) + 1;
  double *mass_bins;
  mass_bins = (double*) my_malloc(nbins*sizeof(double));

  for(int i=0; i<nbins; i++)
    {
      double log10mass = binmin + deltabin*(double)i;
      if (log10mass > binmax) {
	log10mass = binmax;
      }
      double mass = pow(10.,log10mass);
      mass_bins[i] = mass;
      //      printf("mass bin i: %d log10mass: %lf mass: %lf\n", i, log10mass, mass);
    }

  /* create halo structs */
  /* compute density for each halo */
  double ndensB = (double)npointsB/CUBE(Lbox);
  double exp_counts = (4./3.)*M_PI*(CUBE(rmax))*ndensB;
  halo_metadata * halos = my_malloc(npointsA*sizeof(halo_metadata));
  for(i=0;i<npointsA;i++) {
    halos[i].id = halo_id[i];
    halos[i].density = (double)halo_env_counts[i] / exp_counts - 1.0;
    halos[i].mass = halo_mass[i];
  }
  
  my_free(halo_env_counts);
  my_free(halo_id);
  my_free(halo_mass);

  MHash *MassHash = allocate_1d_hash(nbins, mass_bins, npointsA, halos);

  /* sort mass bin and compute percentile ranks */
  sort_1d_hash(MassHash);

  /* copy into linear array, then sort array by halo_id */
  linearize_1d_hash(MassHash, npointsA, halos);
  qsort(halos, npointsA, sizeof(halo_metadata), compare_halo_metadata_by_id);

  /* print halos */
#ifdef PRINT_HALOS
  for(i=0;i<npointsA;i++) {
    printf("id: %lld mass: %f density: %f percentile %f\n",
	   halos[i].id,
	   halos[i].mass,
	   halos[i].density,
	   halos[i].percentile);
  }
#endif

  /* save into (new) HDF5 file */
  write_halo_hdf5(output_filename, "halos", npointsA, halos);

  return 0;
}
