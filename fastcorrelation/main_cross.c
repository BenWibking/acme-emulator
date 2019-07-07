#include "hash.h"

int main(int argc, char *argv[])
{
  /* input: number of points to test, number of cells per axis */
  int nbins;
  double Lbox, minr, maxr;
  int input_nbins, input_njack;
  float input_boxsize, input_rmin, input_rmax;
  char *filenameA, *filenameB;

  /* check inputs */
  if(argc != 8) {
    printf("./auto nbins rmin rmax box_size njackknife_samples filenameA filenameB\n");
    exit(-1);
  }

  input_nbins = atoi(argv[1]);
  input_rmin = atof(argv[2]);
  input_rmax = atof(argv[3]);
  input_boxsize = atof(argv[4]);
  input_njack = atoi(argv[5]);

  filenameA = malloc(sizeof(char)*(strlen(argv[6])+1));
  sprintf(filenameA,"%s",argv[6]);
  filenameB = malloc(sizeof(char)*(strlen(argv[7])+1));
  sprintf(filenameB,"%s",argv[7]);

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
  int ngrid, njack;
  nbins = input_nbins;
  njack = input_njack;
  minr = input_rmin;
  maxr = input_rmax;
  Lbox = (double)input_boxsize;
  /* compute ngrid from rmax */
  ngrid = (int)floor(Lbox/maxr);

  /* read from file */
  particle *pointsA = read_particles_hdf5(filenameA, "particles", &npointsA);
  particle *pointsB = read_particles_hdf5(filenameB, "particles", &npointsB);

  /* generate random points (x,y,z) in unit cube */
  // separate arrays (or Fortran-style arrays) are necessary both for SIMD and cache efficiency
  FLOAT *x1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  FLOAT *y1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));
  FLOAT *z1 = (FLOAT*) my_malloc(npointsA*sizeof(FLOAT));

  FLOAT *x2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  FLOAT *y2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  FLOAT *z2 = (FLOAT*) my_malloc(npointsB*sizeof(FLOAT));
  
  size_t n;
  for(n=0;n<npointsA;n++)
    {
      x1[n] = pointsA[n].x;
      y1[n] = pointsA[n].y;
      z1[n] = pointsA[n].z;
    }
  for(n=0;n<npointsB;n++)
    {
      //      printf("n: %ld x: %f y: %f z: %f\n",n,pointsB[n].x,pointsB[n].y,pointsB[n].z);
      x2[n] = pointsB[n].x;
      y2[n] = pointsB[n].y;
      z2[n] = pointsB[n].z;
    }

  free(pointsA);
  free(pointsB);

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
  uint64_t *pcounts_jackknife = my_malloc(njack*nbins*sizeof(uint64_t));
  uint64_t *pcounts_naive = my_malloc(nbins*sizeof(uint64_t));
  uint64_t *pcounts_jackknife_naive = my_malloc(njack*nbins*sizeof(uint64_t));
  int i;
  for(i=0;i<nbins;i++) {
    pcounts[i] = (long int) 0;
    pcounts_naive[i] = (long int) 0;
  }
  double dlogr = (log10(maxr)-log10(minr))/(double)nbins;
  for(i=0;i<=nbins;i++) {
    double bin_edge = pow(10.0, ((double)i)*dlogr + log10(minr));
    bin_edges[i] = bin_edge;
    bin_edges_sq[i] = SQ(bin_edge);
  }

  cross_count_pairs(grid1, grid2, pcounts, pcounts_jackknife, bin_edges_sq, nbins, njack);

#ifdef TEST_ALL_PAIRS
  cross_count_pairs_naive(x1,y1,z1,npoints1, x2,y2,z2,npoints2, pcounts_naive, pcounts_jackknife, bin_edges_sq, nbins, njack, Lbox);
#endif

  /* output pair counts */
  printf("#min_bin\tmax_bin\tbin_counts\tnatural_estimator\n");

  for(i=0;i<nbins;i++) {
    double ndensA = npointsA/CUBE(Lbox);
    double exp_counts = (4./3.)*M_PI*(CUBE(bin_edges[i+1])-CUBE(bin_edges[i]))*ndensA*npointsB;
    /*double exp_counts_jackknife = exp_counts*(double)(((double)njack-1.0)/(double)njack); */ /* Jackknife */
    double exp_counts_jackknife = exp_counts*(double)(1.0/(double)njack); /* Bootstrap */
    printf("%lf\t%lf\t%ld\t%lf", bin_edges[i],bin_edges[i+1],pcounts[i],(double)pcounts[i]/exp_counts - 1);
    for(int j=0;j<njack;j++) {
      printf("\t%lf",(double)pcounts_jackknife[j*nbins + i]/exp_counts_jackknife - 1);
    }
    printf("\n");
#ifdef TEST_ALL_PAIRS
    printf("(naive) pair counts between (%lf, %lf] = %ld\n",bin_edges[i],bin_edges[i+1],pcounts_naive[i]);
#endif
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
