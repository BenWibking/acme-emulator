#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_result.h>
#include <time.h>

#include "hdf5.h"
#include "hdf5_hl.h"

typedef struct _halo
{
  float mass;
  float X;
  float Y;
  float Z;
  float rvir;
  float rs;
  float weight;
} halo;

typedef struct _HODgal
{
  float X;
  float Y;
  float Z;
  float weight;
  int is_sat;
  float halo_mass;
} HODgal;

typedef struct _halo_metadata
{
  float mass;
  float density;
  float percentile;
} halo_metadata;


void populate_hod(double siglogM, double logMmin, double logM0, double logM1, double alpha, double q_env, double del_gamma, double f_cen, double A_conc, double delta_b, double delta_c, double R_rescale, unsigned long int seed, double Omega_m, double redshift, double Lbox, char *input_fname, char *output_fname, char *env_fname, int is_stochastic);

double NFW_CDF_sampler(float * restrict CDF, gsl_rng *r);

void* read_halo_hdf5(char infile[],char dataset_name[],size_t *len);

herr_t write_gal_hdf5(char filename[], char dataset_name[], size_t len, HODgal* data);

void* read_env_hdf5(char filename[], char dataset_name[], size_t *len);
