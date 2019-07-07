#include "hash.h"

int main(int argc, char *argv[])
{
  /* input: number of points to test, number of cells per axis */
  int nbins;
  double Lbox, minr, maxr;
  int input_nbins, input_npoints;
  float input_boxsize, input_rmin, input_rmax, input_nsubsamples;

  /* check inputs */
  if(argc != 7) {
    printf("./auto nbins rmin rmax box_size nsubsamples npoints\n");
    exit(-1);
  }

  input_nbins = atoi(argv[1]);
  input_rmin = atof(argv[2]);
  input_rmax = atof(argv[3]);
  input_boxsize = atof(argv[4]);
  input_nsubsamples = atoi(argv[5]);
  input_npoints = atoi(argv[6]);

  if(input_npoints <= 0) {
    printf("npoints must be positive!\n");
    exit(-1);
  }

  if(input_nbins <= 0) {
    printf("ngrid must be positive!\n");
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
  if(input_nsubsamples < 0) {
    printf("nsubsamples must be nonnegative!\n");
    exit(-1);
  }
  if(pow(floor(pow((float)input_nsubsamples, 1./3.)), 3) != input_nsubsamples) {
    printf("nsubsamples must be a perfect cube!\n");
    exit(-1);
  }

  size_t npoints;
  int ngrid, nsubsamples, nsubsamples_along_side;
  nbins = input_nbins;
  nsubsamples = input_nsubsamples;
  nsubsamples_along_side = pow((double)nsubsamples, 1./3.);
  minr = input_rmin;
  maxr = input_rmax;
  Lbox = (double)input_boxsize;
  /* compute ngrid from rmax */
  ngrid = (int)floor(Lbox/maxr);
  npoints = input_npoints;

  // separate arrays (or Fortran-style arrays) are necessary both for SIMD and cache efficiency
  FLOAT *x = (FLOAT*) my_malloc(npoints*sizeof(FLOAT));
  FLOAT *y = (FLOAT*) my_malloc(npoints*sizeof(FLOAT));
  FLOAT *z = (FLOAT*) my_malloc(npoints*sizeof(FLOAT));

  grid_id *label;
  label = my_malloc(npoints*sizeof(grid_id));

  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  int seed = 42;
  gsl_rng_set(r, seed); /* Seeding random distribution */

  for(size_t n=0;n<npoints;n++)
    {
      x[n] = gsl_rng_uniform(r)*Lbox;
      y[n] = gsl_rng_uniform(r)*Lbox;
      z[n] = gsl_rng_uniform(r)*Lbox;

      int jx = (int)floor(x[n]/Lbox*((double)nsubsamples_along_side)) % nsubsamples_along_side;
      int jy = (int)floor(y[n]/Lbox*((double)nsubsamples_along_side)) % nsubsamples_along_side;
      int jz = (int)floor(z[n]/Lbox*((double)nsubsamples_along_side)) % nsubsamples_along_side;
      label[n].x = jx;
      label[n].y = jy;
      label[n].z = jz;
    }

  /* hash into grid cells */
  GHash *grid = allocate_hash(ngrid, nsubsamples, Lbox, npoints, x, y, z);
  if ((int)grid == 0) {
    printf("allocating grid failed!\n");
    exit(-1);
  }

  fprintf(stderr,"computing geometric hash...");
  geometric_hash(grid, x, y, z, npoints);
  fprintf(stderr,"done!\n");

  /* compute pair counts assuming periodic box */
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

  fprintf(stderr,"computing pair counts...");
  count_pairs(grid, pcounts, pcounts_jackknife, bin_edges_sq, nbins);
  fprintf(stderr,"done!\n");

  count_pairs_naive(x,y,z, label, npoints, pcounts_naive, pcounts_jackknife_naive, \
		    bin_edges_sq, nbins, nsubsamples, Lbox);

  /* output pair counts */
  printf("min_bin\tmax_bin\tbin_counts\tnatural_estimator\n");
  int count_bug = 0;
  for(int i=0;i<nbins;i++) {
    /* test pair counts */
    if (pcounts[i] != pcounts_naive[i]) {
      printf("\n*** BUG DETECTED in counts! ***\n");
      count_bug++;
    }

    double ndens = npoints/CUBE(Lbox);
    double exp_counts = (4./3.)*M_PI*(CUBE(bin_edges[i+1])-CUBE(bin_edges[i]))*ndens*npoints;
    double exp_counts_jackknife = exp_counts*(double)((1.0)/(double)nsubsamples);
    printf("%lf\t%lf\t%ld\t%lf",bin_edges[i],bin_edges[i+1],pcounts[i],(double)pcounts[i]/exp_counts - 1.0);
    for(int j=0;j<nsubsamples;j++) {
      printf("\t%lf",(double)pcounts_jackknife[j*nbins + i]/exp_counts_jackknife - 1.0);
    }
    printf("\n");

    printf("%lf\t%lf\t%ld\t%lf",bin_edges[i],bin_edges[i+1],pcounts_naive[i],(double)pcounts_naive[i]/exp_counts - 1.0);
    for(int j=0;j<nsubsamples;j++) {
      printf("\t%lf",(double)pcounts_jackknife_naive[j*nbins + i]/exp_counts_jackknife - 1.0);
    }
    printf("\n");

    printf("\n");
  }

  /* free memory */
  my_free(pcounts);
  my_free(bin_edges);
  my_free(bin_edges_sq);

  my_free(x);
  my_free(y);
  my_free(z);
  
  free_hash(grid);
 
  /* return 1 to signal test failure */
  if(count_bug > 0) {
    return 1;
  }

  return 0;
}
