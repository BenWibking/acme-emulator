#include "hash.h"

int main(int argc, char *argv[])
{
  /* input: number of points to test, number of cells per axis */
  int nbins;
  double Lbox, minr, maxr;
  int input_nbins;
  float input_boxsize, input_rmin, input_rmax, input_njackknife;
  char *filename;
  bool do_test_pairs = false; /* if true, do N^2 pair counts for debugging */

  /* check inputs */
  if(argc != 7 && argc != 8) {
    printf("./auto [--test-pairs] nbins rmin rmax box_size njackknife_samples filename\n");
    exit(-1);
  }

  int countargs = 1;
  if(argc == 8) {
    if(strcmp("--test-pairs",argv[countargs])==0) {
      fprintf(stderr,"***RUNNING CORRECTNESS TESTS (THIS IS EXPENSIVE)***\n");
      do_test_pairs = true;
    }
    countargs++;
  }

  input_nbins = atoi(argv[countargs]); countargs++;
  input_rmin = atof(argv[countargs]); countargs++;
  input_rmax = atof(argv[countargs]); countargs++;
  input_boxsize = atof(argv[countargs]); countargs++;
  input_njackknife = atoi(argv[countargs]); countargs++;

  filename = malloc(sizeof(char)*(strlen(argv[countargs])+1));
  if(filename) { /* filename is not null */
    sprintf(filename,"%s",argv[countargs]);
  } else {
    fprintf(stderr,"malloc failure! cannot allocate filename array\n");
    exit(-1);
  }

  if(input_nbins <= 0) {
    fprintf(stderr,"ngrid must be positive!\n");
    exit(-1);
  }
  if(input_rmin <= 0.) {
    fprintf(stderr,"rmin must be positive!\n");
    exit(-1);
  }
  if(!(input_rmax > input_rmin)) {
    fprintf(stderr,"rmax must be greater than rmin!\n");
    exit(-1);
  }
  if(input_boxsize <= 0.) {
    fprintf(stderr,"boxsize must be positive!\n");
    exit(-1);
  }
  if(input_njackknife < 0) {
    fprintf(stderr,"njackknife_samples must be nonnegative!\n");
    exit(-1);
  }
  if(pow(floor(pow((float)input_njackknife, 1./3.)), 3) != input_njackknife) {
    fprintf(stderr,"njackknife_samples must be a perfect cube!\n");
    exit(-1);
  }

  size_t npoints;
  int ngrid, njack, nsubsamples_along_side;
  nbins = input_nbins;
  njack = input_njackknife;
  nsubsamples_along_side = pow((double)njack, 1./3.);
  minr = input_rmin;
  maxr = input_rmax;
  Lbox = (double)input_boxsize;
  /* compute ngrid from rmax */
  ngrid = (int)floor(Lbox/maxr);

  /* read from file */
  particle *points = read_particles_hdf5(filename, "particles", &npoints);

  // separate arrays (or Fortran-style arrays) are necessary both for SIMD and cache efficiency
  FLOAT *x = (FLOAT*) my_malloc(npoints*sizeof(FLOAT));
  FLOAT *y = (FLOAT*) my_malloc(npoints*sizeof(FLOAT));
  FLOAT *z = (FLOAT*) my_malloc(npoints*sizeof(FLOAT));

  grid_id *label;
  size_t n;
  for(n=0;n<npoints;n++)
    {
      x[n] = points[n].x;
      y[n] = points[n].y;
      z[n] = points[n].z;      
    }

  if(do_test_pairs) {
    label = my_malloc(npoints*sizeof(grid_id));
    for(n=0;n<npoints;n++)
      {
	int jx = (int)floor(x[n]/Lbox*((double)nsubsamples_along_side)) % nsubsamples_along_side;
	int jy = (int)floor(y[n]/Lbox*((double)nsubsamples_along_side)) % nsubsamples_along_side;
	int jz = (int)floor(z[n]/Lbox*((double)nsubsamples_along_side)) % nsubsamples_along_side;
	label[n].x = jx;
	label[n].y = jy;
	label[n].z = jz;
      }
  }

  if(!do_test_pairs) {
    free(points);
  }

  /* hash into grid cells */
  GHash *grid = allocate_hash(ngrid, njack, Lbox, npoints, x, y, z);
  if ((int)grid == 0) {
    fprintf(stderr,"allocating grid failed!\n");
    exit(-1);
  }

  geometric_hash(grid, x, y, z, npoints);

  /* compute pair counts assuming periodic box */
  double *bin_edges_sq = my_malloc((nbins+1)*sizeof(double));
  double *bin_edges = my_malloc((nbins+1)*sizeof(double));
  uint64_t *pcounts = my_malloc(nbins*sizeof(uint64_t));
  uint64_t *pcounts_jackknife = my_malloc(njack*nbins*sizeof(uint64_t));
  uint64_t *pcounts_naive = my_malloc(nbins*sizeof(uint64_t));
  uint64_t *pcounts_jackknife_naive = my_malloc(njack*nbins*sizeof(uint64_t));
  int i;
  for(i=0;i<nbins;i++) {
    pcounts[i] = (int64_t) 0;
    pcounts_naive[i] = (int64_t) 0;
  }
  for(int i=0;i<njack;i++) {
    for(int j=0;j<nbins;j++) {
      pcounts_jackknife[i*nbins + j] = (int64_t) 0;
    }
  }
  double dlogr = (log10(maxr)-log10(minr))/(double)nbins;
  for(i=0;i<=nbins;i++) {
    double bin_edge = pow(10.0, ((double)i)*dlogr + log10(minr));
    bin_edges[i] = bin_edge;
    bin_edges_sq[i] = SQ(bin_edge);
  }

  count_pairs(grid, pcounts, pcounts_jackknife, bin_edges_sq, nbins);

  if(do_test_pairs) { /* test pair counts */
    fprintf(stderr,"computing N^2 pair counts...");
    count_pairs_naive(x,y,z, label, npoints, pcounts_naive, pcounts_jackknife, bin_edges_sq, nbins, njack, Lbox);
    fprintf(stderr,"done!\n");
  }

  /* output pair counts */
  printf("#min_bin\tmax_bin\tbin_counts\tnatural_estimator\n");

  int bug = 0;
  for(i=0;i<nbins;i++) {
    double ndens = npoints/CUBE(Lbox);
    double exp_counts = (4./3.)*M_PI*(CUBE(bin_edges[i+1])-CUBE(bin_edges[i]))*ndens*npoints;
#ifdef JACKKNIFE_SUBSAMPLES
    double exp_counts_jackknife = exp_counts*(double)(((double)njack-1.0)/(double)njack);
#else /* bootstrap subsamples */
    double exp_counts_jackknife = exp_counts*(double)(1.0/(double)njack);
#endif
    printf("%lf\t%lf\t%ld\t%lf", bin_edges[i],bin_edges[i+1],pcounts[i],(double)pcounts[i]/exp_counts - 1.0);
    for(int j=0;j<njack;j++) {
      printf("\t%lf",(double)pcounts_jackknife[j*nbins + i]/exp_counts_jackknife - 1.0);
    }
    printf("\n");

    if(do_test_pairs) { /* test pair counts */
      if(pcounts[i] != pcounts_naive[i]) {
	fprintf(stderr,"\n*** BUG DETECTED in counts! ***\n");
	bug++;
      }
      printf("(naive) pair counts = %ld\n",pcounts_naive[i]);
    }
  }

  /* free memory */
  my_free(pcounts);
  my_free(pcounts_jackknife);
  my_free(bin_edges);
  my_free(bin_edges_sq);

  my_free(x);
  my_free(y);
  my_free(z);
  
  free_hash(grid);

  /* if --test-pairs and we get wrong results, return error */
  if(bug > 0) {
    return 1;
  }

  return 0;
}
