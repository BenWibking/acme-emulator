#include "read_hdf5.h"

#define M_PI 3.14159265358979323846
#define Mpc_to_cm 3.0856e24		/*Conversion factor from Mpc to cm */
#define Msun_to_g 1.989e33		/*Conversion factor from Msun to grams*/
#define G 6.672e-8				/*Universal Gravitational Constant in cgs units*/
#define Hubble 3.2407789e-18	/*Hubble's constant h/sec*/

/*Cosmological Critical Density in Msun h^2 / Mpc^3 */
#define rho_crit (3.0*pow(Hubble, 2.0) / (8.0 * M_PI * G)) * (pow(Mpc_to_cm, 3.0) / Msun_to_g) 

#define INDEX4(i,j) (i*4 + j)


/*	These functions populate halos with HOD galaxies from an HDF5 halo catalog
		The outputs specify galaxy positions, and velocities
		(host mass and satellite/central identification have been dropped
		for speed and space considerations)
		For now just positions	*/


float central_weight
(float logM, float env_rank, float f_logMmin_0, float f_siglogM, float f_q_env)
{

	/* compute modified logMmin for this halo's environment */
	float f_logMmin = f_logMmin_0 + f_q_env * (env_rank - 0.5f);
	
	/*Mean central occupation or the probability of hosting a central*/
	float prob = 0.5 * (1.0 + erf( (logM - f_logMmin) / f_siglogM) );

	return prob;
	
}


halo * find_galaxy_hosts
(halo halos[], halo_metadata * env, double siglogM, double logMmin, double q_env,
 double f_cen, unsigned long int N_h, unsigned long int *Ncen, int stochastic, gsl_rng *r)
{

	/* This function uses the Zehavi 2011 prescription
		 to find the halos that host central galaxies.
		 Modified to use the weighting scheme of Guo & Zheng 2015. */

	uint64_t i;
	uint64_t j = 0;

	float f_logMmin_0 = (float)logMmin;
	float f_siglogM = (float)siglogM;
	float f_q_env = (float)q_env;

	halo *host_coords = malloc(N_h * sizeof(halo));

	for(i = 0; i < N_h; i++)
	{
		
		float logM = (float)log10(halos[i].mass);
		float env_rank = env[i].percentile;
		float prob = f_cen * central_weight(logM, env_rank, f_logMmin_0, f_siglogM, f_q_env);
	
		if(stochastic == 1) {
			/* draw from RNG for rejection sampling */
			double sample = gsl_rng_uniform(r);
	
			if(prob < sample) {  
		 		continue; /* reject */
			}
		}
	
		host_coords[j].X = halos[i].X;
		host_coords[j].Y = halos[i].Y;
		host_coords[j].Z = halos[i].Z;
		host_coords[j].mass = halos[i].mass;
		host_coords[j].rvir = halos[i].rvir;
		host_coords[j].rs = halos[i].rs;
	
		if(stochastic == 1) {
			host_coords[j].weight = 1.0;
		} else {
			host_coords[j].weight = prob;
		}
		
		j++;

	}

	*Ncen = j;

	return host_coords;
	
}


float * find_satellites
(halo halos[], halo_metadata * env, double siglogM, double logMmin, double q_env,
 double logM0, double logM1, double alpha, unsigned long int N_h)
{

	/*This function determines how many satellite galaxies each halo has
		using the same Zehavi 2011 prescription*/

	float * mean_satellites = malloc(N_h*sizeof(float));
	uint64_t i;
	
	for(i=0; i < N_h; i++) 
	{
		double M0 = pow(10.0, logM0);
		double M1 = pow(10.0, logM1);
		
		float logM = log10(halos[i].mass);
		
		// should not and does not include f_cen here
		double mean_cen = central_weight(logM, env[i].percentile, logMmin, siglogM, q_env);
		double mean_sat;
	
		if(logM < logM0) {
		  mean_sat = 0.0; /* Enforcing the satellite cutoff */
		} else {
		  mean_sat = mean_cen * pow( ( ( halos[i].mass - M0 ) / M1 ), alpha );
		}
	
		mean_satellites[i] = mean_sat;
	}
	
	return mean_satellites;
	
}


HODgal * pick_NFW_satellites
(halo host, const int N_sat, double O_m, double z, double del_gamma, double A_conc,
 double delta_b, double delta_c, double R_rescale, gsl_rng *r)
{

	/* This function determines the spatial distribution of the satellite galaxies
		Galaxies are NFW-distributed using results from Correa et al. 2015
		We allow for a deviation from NFW in the galaxy distribution
			by a factor of r^delta_gamma.
		Concentration-mass relation can be modified by A_conc (amplitude)
			or delta_b (power-law slope).
		Halo radii are rescaled by the factor R_rescale. */

	HODgal * coords = malloc(N_sat * sizeof(HODgal));

	double logM = log10(host.mass);
	double x0 = host.X;
	double y0 = host.Y;
	double z0 = host.Z;

	
	/* Concentration-mass relation from Correa et al. 2015 */
	
	// double alpha = 1.62774 - 0.2458*(1.0 + z) + 0.01716*pow(1.0 + z, 2.0);
	// double beta = 1.66079 + 0.00359*(1.0 + z) - 1.6901*pow(1.0 + z, 0.00417);
	// double gamma = -0.02049 + 0.0253*pow(1.0 + z ,-0.1044);
	// beta += delta_b;
	// gamma += delta_c;
	// double D_vir = 200.0, rho_u = rho_crit*O_m;
	// double exponent = alpha + beta*logM*(1.0 + gamma*pow(logM, 2.0)); 	
	/* Approximate factor (\sqrt 2) to rescale Rvir between crit, matter */
	// double cvir = A_conc * sqrt(2.0) * pow(10.0, exponent); 
	// double R_vir = R_rescale * pow((3.0/4.0)*(1.0/M_PI)*(host.mass/(D_vir*rho_u)), 1.0/3.0);
	
	
	/* Use catalog Rvir, rs.  They are in *kpc/h* */
	
	double R_vir = R_rescale * ((double) host.rvir) / 1000.0;	// Mpc/h (comoving)
	double rs = ((double) host.rs) / 1000.0;					// Mpc/h (comoving)
	double cvir = A_conc * (R_vir / rs);						// dimensionless


	/* Compute profiles */
	
	float CDF[1000];
	int j;
	size_t i;
	gsl_sf_result hyperg_result;
	int status = gsl_sf_hyperg_2F1_e(1.0, 1.0,
									 3.0 + del_gamma, cvir/(1.0 + cvir), &hyperg_result);
	
	/* check return code */
	
	if(status) {
		fprintf(stderr, "failed at prefac, error: %s\n", gsl_strerror(status));
		//fprintf(stderr, "inputs: del_gamma = %f \t cvir = %f\n",
		//					del_gamma, cvir);
		//fprintf(stderr, "inputs: R_vir = %f \t logM = %f \t exponent = %f\n",
		//					R_vir, logM, exponent);
		//fprintf(stderr, "inputs: alpha = %f \t beta = %f \t gamma = %f\n",
		//					alpha, beta, gamma);
		exit(-1);
	}
	
	// double prefac = 1.0/( log( 1.0 + cvir ) - (cvir / ( 1.0 + cvir )) );
	double prefac = (1.0 + cvir) / ( 2.0 + del_gamma - (1.0 + del_gamma)*(hyperg_result.val));
	
	float f_c_vir = (float)cvir;

	/* TODO: de-dimensionalize in terms of r_s.. this makes the profile universal... */
	
	for(i=0; i<1000; i++)
	{
	
		float x = (float)i / 1000.0;
		
		/* CDF[i] = prefac * \
					( log( 1.0 + x * f_c_vir ) - (x * f_c_vir / ( 1.0 + x*f_c_vir )) ); */
					
		int status = gsl_sf_hyperg_2F1_e(1.0, 1.0, 3.0 + del_gamma,
										 x*cvir/(1.0 + x*cvir), &hyperg_result);
		
		if(status) {
			fprintf(stderr, "failed at CDF, error: %s\n", gsl_strerror(status));
			fprintf(stderr, "inputs: del_gamma = %f \t cvir = %f\n", del_gamma, cvir);
			exit(-1);
		}
		
		CDF[i] = prefac * pow( x, (2.0 + del_gamma) ) * \
				 ( 2.0 + del_gamma - (1.0 + del_gamma)*(hyperg_result.val) ) / (1 + cvir*x);
		
	}
	
	for(j=0; j<N_sat; j++)
	{
	
		double frac = NFW_CDF_sampler(&CDF[0], r);
		double R = R_vir * frac;
		
		/* Sphere point picking */
		double phi = 2.0*M_PI*gsl_rng_uniform(r), costheta = 2.0*gsl_rng_uniform(r) - 1.0;
		double sintheta = sqrt(1.0 - costheta*costheta);
		
		/* Satellite Galaxy Coordinates */
		double x = R*sintheta*cos(phi)+x0;
		double y = R*sintheta*sin(phi)+y0;
		double z = R*costheta+z0;
		coords[j].X = x;
		coords[j].Y = y;
		coords[j].Z = z;
		
	}

	return coords;
	
}


double wrap_periodic(double x, double Lbox)
{
	if((x < Lbox) && (x >= 0.)) {
		return x;
	} else if (x >= Lbox) {	// x > Lbox
		return (x - Lbox);
	} else {				// x < 0.0
		return (x + Lbox);
	}
}


void populate_hod
(double siglogM, double logMmin, double logM0, double logM1, double alpha, double q_env,
 double del_gamma, double f_cen, double A_conc, double delta_b, double delta_c,
 double R_rescale, unsigned long int seed, double Omega_m, double redshift,
 double Lbox, char *input_fname, char *output_fname, char *env_fname, int stochastic)
{

	#define OVERSAMPLE_FACTOR 10

	herr_t status;
	size_t NumData,i;
	halo *data;

	uint64_t Ncen, Nsat;

	const gsl_rng_type * T;
	gsl_rng * r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r, seed); /* Seeding random distribution */
	gsl_set_error_handler_off();

	data = read_halo_hdf5(input_fname,"halos",&NumData);

	halo_metadata *env;
	size_t NumEnv;
	env = read_env_hdf5(env_fname,"halos",&NumEnv);

	/* check that NumData == NumEnv */
	if(!(NumEnv == NumData)) exit(1);
	uint64_t Nhalos = NumData;

	/* compute HOD parameters from number density, mass function, environment density */  

	halo *cenhalos; //Central Coordinates
	cenhalos = find_galaxy_hosts(data, env, siglogM, logMmin, q_env, f_cen,
			       Nhalos, &Ncen, stochastic, r);
			       
	HODgal * cens = malloc(Ncen*sizeof(HODgal));

	for(i=0; i<Ncen; i++){
		cens[i].X = cenhalos[i].X;
		cens[i].Y = cenhalos[i].Y;
		cens[i].Z = cenhalos[i].Z;
		cens[i].weight = cenhalos[i].weight;
		cens[i].halo_mass = cenhalos[i].mass;
	}

	float *mean_sats; //mean number of satellites for each halo
	
	mean_sats = find_satellites(data, env, siglogM, logMmin, q_env,
								logM0, logM1, alpha, Nhalos);
								
	Nsat = 0;
	unsigned int *nsats = malloc(Nhalos*sizeof(unsigned int));

	int j,k,l=0;
	for(j=0;j<Nhalos;j++)
	{
	
		if(stochastic == 1) {
			/* sample from Poisson */
			if(mean_sats[j] > 0.) {
				nsats[j] = gsl_ran_poisson(r, mean_sats[j]);
				Nsat += nsats[j];
			} else {
				nsats[j] = 0;
			}
			
		} else {
			if(mean_sats[j] > 0.) {
				nsats[j] = OVERSAMPLE_FACTOR * (int)ceil(mean_sats[j]);
				Nsat += nsats[j];
			} else {
				nsats[j] = 0;
			}
		}
		
	}
	
	// Satellite Coordinates
	HODgal * coords  = malloc(Nsat*sizeof(HODgal)); 

	for(j=0;j<Nhalos;j++)
	{
		if(nsats[j] > 0){
		
			HODgal * halosats = malloc(nsats[j] * sizeof(HODgal));
			halosats = pick_NFW_satellites(data[j], nsats[j], Omega_m, redshift,
					     				   del_gamma, A_conc, delta_b, delta_c, R_rescale, r);
					     				   
			for(k=0; k<nsats[j]; k++)
			{
				coords[l].X = wrap_periodic(halosats[k].X,Lbox);
			 	coords[l].Y = wrap_periodic(halosats[k].Y,Lbox);
			 	coords[l].Z = wrap_periodic(halosats[k].Z,Lbox);
			  
			 	if(stochastic == 1) {
			  		coords[l].weight = 1.0;
			 	} else {
			 		coords[l].weight = (mean_sats[j] / (float)nsats[j]);
			 	}
		
		  		coords[l].halo_mass = data[j].mass;
		  		l++;
			}
			
			free(halosats);
		}
	}
	
	free(cenhalos);
	free(mean_sats);

	uint64_t len = Nsat + Ncen;

	HODgal *HODgals = malloc(len*sizeof(HODgal));

	for(i=0; i<Ncen;i++)
	{
		HODgals[i].X = cens[i].X;
		HODgals[i].Y = cens[i].Y;
		HODgals[i].Z = cens[i].Z;
		HODgals[i].weight = cens[i].weight;
		HODgals[i].is_sat = 0;
		HODgals[i].halo_mass = cens[i].halo_mass;
	}
	
	free(cens);
	
	for(i=0; i<Nsat; i++)
	{
		HODgals[i+Ncen].X = coords[i].X;
		HODgals[i+Ncen].Y = coords[i].Y;
		HODgals[i+Ncen].Z = coords[i].Z;
		HODgals[i+Ncen].weight = coords[i].weight;
		HODgals[i+Ncen].is_sat = 1;
		HODgals[i+Ncen].halo_mass = coords[i].halo_mass;
	}
	
	free(coords);
	
	status = write_gal_hdf5(output_fname, "particles", (size_t)len, HODgals);
	
	free(HODgals);
	
	gsl_rng_free(r);
	
}
