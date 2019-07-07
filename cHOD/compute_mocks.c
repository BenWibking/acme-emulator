#include "read_hdf5.h"

int main(int argc, char *argv[])
{
  if(argc != 21) {
    printf("%d usage: ./compute_mocks Omega_m redshift siglogM logMmin logM0 logM1 alpha q_env del_gamma f_cen A_conc delta_b delta_c R_rescale boxsize [halo catalog file] [galaxy mock file] [halo environment file] is_stochastic seed\n",argc);
    return -1;
  }

  double Omega_m = strtod(argv[1], NULL);
  double redshift = strtod(argv[2], NULL);

  double siglogM = strtod(argv[3], NULL);
  double logMmin = strtod(argv[4], NULL);
  double logM0 = strtod(argv[5], NULL);
  double logM1 = strtod(argv[6], NULL);
  double alpha = strtod(argv[7], NULL);
  double q_env = strtod(argv[8], NULL);
  double del_gamma = strtod(argv[9], NULL);
  double f_cen = strtod(argv[10], NULL);
  double A_conc = strtod(argv[11], NULL);
  double delta_b = strtod(argv[12], NULL);
  double delta_c = strtod(argv[13], NULL);
  double R_rescale = strtod(argv[14], NULL);

  double boxsize = strtod(argv[15], NULL);

  char *halo_file, *output_file, *env_file;
  size_t halo_file_ssize, output_file_ssize, env_file_ssize;

  halo_file_ssize = sizeof(char)*(strlen(argv[16]) + 1);
  output_file_ssize = sizeof(char)*(strlen(argv[17]) +1 );
  env_file_ssize = sizeof(char)*(strlen(argv[18]) +1 );

  int is_stochastic = atoi(argv[19]);
  int seed = atoi(argv[20]);
  //  int seed = 42;

  halo_file = malloc(halo_file_ssize);
  output_file = malloc(output_file_ssize);
  env_file = malloc(env_file_ssize);
  snprintf(halo_file, halo_file_ssize, "%s", argv[16]);
  snprintf(output_file, output_file_ssize, "%s", argv[17]);
  snprintf(env_file, env_file_ssize, "%s", argv[18]);

  //  fprintf(stderr,"Computing HOD from %s\n", halo_file);
  //  fprintf(stderr,"Reading environment density from %s\n", env_file);
  //  fprintf(stderr,"Saving to output file %s\n", output_file);
  //  fprintf(stderr,"is_stochastic = %i; random seed = %i\n",is_stochastic,seed);

  populate_hod(siglogM, logMmin, logM0, logM1, alpha, q_env, del_gamma, f_cen, A_conc, delta_b, delta_c, R_rescale, seed, Omega_m, redshift, boxsize, halo_file, output_file, env_file, is_stochastic);

  return 0;
}
