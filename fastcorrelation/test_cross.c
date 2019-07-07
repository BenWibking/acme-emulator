#include "hash.h"

int main(int argc, char *argv[])
{
  /* input: number of points to test, number of cells per axis */
  int nbins;
  double Lbox, minr, maxr;
  int input_nbins, input_npointsA, input_npointsB, input_njack;
  float input_boxsize, input_rmin, input_rmax;

  /* check inputs */
  if(argc != 8) {
    printf("./auto nbins rmin rmax box_size njackknife_samples npointsA npointsB\n");
    exit(-1);
  }

  input_nbins = atoi(argv[1]);
  input_rmin = atof(argv[2]);
  input_rmax = atof(argv[3]);
  input_boxsize = atof(argv[4]);
  input_njack = atoi(argv[5]);
  input_npointsA = atoi(argv[6]);
  input_npointsB = atoi(argv[7]);

  if(input_npointsA <= 0) {
    printf("npointsA must be positive!\n");
    exit(-1);
  }

  if(input_npointsB <= 0) {
    printf("npointsB must be positive!\n");
    exit(-1);
  }

  if(input_nbins <= 0) {
    printf("ngrid must be positive!\n");
    exit(-1);
  }

  if(input_njack <= 0) {
    printf("njackknife_samples must be positive!\n");
    exit(-1);
  }

  if(input_rmin <= 0.) {
    printf("rmin must be positive!\n");
    exit(-1);
  }
  if(!(input_rmax > input_rmin)) {
    printf("rmax must be greater than rmin!\n");
    exit(-1);
  }
  if(input_boxsize <= 0.) {
    printf("boxsize must be positive!\n");
    exit(-1);
  }

  size_t npointsA,npointsB;
  int ngrid, njack, nsamples_along_side;
  njack = input_njack;
  nsamples_along_side = (int) pow((double)njack,1./3.);
  nbins = input_nbins;
  minr = input_rmin;
  maxr = input_rmax;
  Lbox = (double)input_boxsize;
  /* compute ngrid from rmax */
  ngrid = (int)floor(Lbox/maxr);
  npointsA = input_npointsA;
  npointsB = input_npointsB;

  /* generate random points (x,y,z) in unit cube */
  // separate arrays (or Fortran-style arrays) are necessary both for SIMD and cache efficiency
  FLOAT *x1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  FLOAT *y1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  FLOAT *z1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));

  FLOAT *x2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  FLOAT *y2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  FLOAT *z2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));

  grid_id *label1, *label2;
  label1 = my_malloc(npointsA*sizeof(grid_id));
  label2 = my_malloc(npointsB*sizeof(grid_id));

  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  int seed = 42;
  gsl_rng_set(r, seed); /* Seeding random distribution */

  size_t n;
  for(n=0;n<npointsA;n++)
    {
      x1[n] = gsl_rng_uniform(r)*Lbox;
      y1[n] = gsl_rng_uniform(r)*Lbox;
      z1[n] = gsl_rng_uniform(r)*Lbox;

      /* fix variable naming convention - nsubsamples vs. nsubsamples_along_side */
      int jx = (int)floor(x1[n]/Lbox*((double)nsamples_along_side)) % nsamples_along_side;
      int jy = (int)floor(y1[n]/Lbox*((double)nsamples_along_side)) % nsamples_along_side;
      int jz = (int)floor(z1[n]/Lbox*((double)nsamples_along_side)) % nsamples_along_side;
      label1[n].x = jx;
      label1[n].y = jy;
      label1[n].z = jz;
    }
  for(n=0;n<npointsB;n++)
    {
      x2[n] = gsl_rng_uniform(r)*Lbox;
      y2[n] = gsl_rng_uniform(r)*Lbox;
      z2[n] = gsl_rng_uniform(r)*Lbox;

      /* fix variable naming convention - nsubsamples vs. nsubsamples_along_side */
      int jx = (int)floor(x2[n]/Lbox*((double)nsamples_along_side)) % nsamples_along_side;
      int jy = (int)floor(y2[n]/Lbox*((double)nsamples_along_side)) % nsamples_along_side;
      int jz = (int)floor(z2[n]/Lbox*((double)nsamples_along_side)) % nsamples_along_side;
      label2[n].x = jx;
      label2[n].y = jy;
      label2[n].z = jz;
    }

  gsl_rng_free(r);
  
  /* hash into grid cells */
  GHash *grid1 = allocate_hash(ngrid, njack, Lbox, npointsA, x1, y1, z1);
  GHash *grid2 = allocate_hash(ngrid, njack, Lbox, npointsB, x2, y2, z2);
  if ((int)grid1 == 0) {
    printf("allocating grid1 failed!\n");
    exit(-1);
  }
  if ((int)grid2 == 0) {
    printf("allocating grid2 failed!\n");
    exit(-1);
  }

  geometric_hash(grid1, x1, y1, z1, npointsA);
  geometric_hash(grid2, x2, y2, z2, npointsB);
  

  /* compute pair counts */
  double *bin_edges_sq = my_malloc((nbins+1)*sizeof(double));
  double *bin_edges = my_malloc((nbins+1)*sizeof(double));
  uint64_t *pcounts = my_malloc(nbins*sizeof(uint64_t));
  uint64_t *pcounts_naive = my_malloc(nbins*sizeof(uint64_t));
  uint64_t *pcounts_jackknife = my_malloc(nbins*sizeof(uint64_t));
  uint64_t *pcounts_jackknife_naive = my_malloc(nbins*sizeof(uint64_t));

  for(int i=0;i<nbins;i++) {
    pcounts[i] = (uint64_t) 0;
    pcounts_naive[i] = (uint64_t) 0;
  }

  double dlogr = (log10(maxr)-log10(minr))/(double)nbins;
  for(int i=0;i<=nbins;i++) {
    double bin_edge = pow(10.0, ((double)i)*dlogr + log10(minr));
    bin_edges[i] = bin_edge;
    bin_edges_sq[i] = SQ(bin_edge);
  }

  cross_count_pairs(grid1, grid2, pcounts, pcounts_jackknife, bin_edges_sq, nbins, njack);

  cross_count_pairs_naive(x1,y1,z1,label1,npointsA,\
			  x2,y2,z2,label2,npointsB,\
			  pcounts_naive,pcounts_jackknife,bin_edges_sq, nbins, njack, Lbox);

  /* output pair counts */
  printf("min_bin\tmax_bin\tbin_counts\tnatural_estimator\n");
  for(int i=0;i<nbins;i++) {
    double ndensA = npointsA/CUBE(Lbox);
    double exp_counts = (4./3.)*M_PI*(CUBE(bin_edges[i+1])-CUBE(bin_edges[i]))*ndensA*npointsB;
    double exp_counts_jackknife = exp_counts*((double)1.0/(double)njack); /* bootstrap */
    printf("%lf\t%lf\t%ld\t%lf",bin_edges[i],bin_edges[i+1],pcounts[i],(double)pcounts[i]/exp_counts - 1.0);
    for(int j=0;j<njack;j++) {
      printf("\t%lf",(double)pcounts_jackknife[j*nbins + i]/exp_counts_jackknife - 1.0);
    }
    printf("\n");

    printf("%lf\t%lf\t%ld\t%lf",bin_edges[i],bin_edges[i+1],pcounts_naive[i],(double)pcounts_naive[i]/exp_counts - 1.0);
    for(int j=0;j<njack;j++) {
      printf("\t%lf",(double)pcounts_jackknife_naive[j*nbins + i]/exp_counts_jackknife - 1.0);
    }
    printf("\n");

    printf("\n");
  }

  my_free(pcounts);
  my_free(bin_edges);
  my_free(bin_edges_sq);
  
  my_free(x1);
  my_free(y1);
  my_free(z1);
  my_free(x2);
  my_free(y2);
  my_free(z2);

  free_hash(grid1);
  free_hash(grid2);
  return 0;
}
