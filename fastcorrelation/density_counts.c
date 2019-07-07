#include "hash.h"

void density_count_pairs(GHash * restrict g1, GHash * restrict g2, uint64_t * halo_env_counts, uint64_t * halo_id, FLOAT * halo_mass, double rmax_sq)
{
  /* check that g1 and g2 have the same ngrid and Lbox */
  if(g1->ngrid != g2->ngrid) {
    printf("grids do not align!\n");
    exit(-1);
  }
  if(g1->Lbox != g2->Lbox) {
    printf("box geometries do not align!\n");
    exit(-1);
  }

  int ngrid = g1->ngrid;
  double Lbox = g1->Lbox;

  int ix,iy,iz;
  size_t q = 0;
  for(ix=0;ix<ngrid;ix++) {
    for(iy=0;iy<ngrid;iy++) {
      for(iz=0;iz<ngrid;iz++) {

	/* do this for each cell */
	size_t count1 = g1->counts[INDEX(ix,iy,iz)];
	FLOAT * restrict x1 = g1->x[INDEX(ix,iy,iz)];
	FLOAT * restrict y1 = g1->y[INDEX(ix,iy,iz)];
	FLOAT * restrict z1 = g1->z[INDEX(ix,iy,iz)];
	FLOAT * restrict mass = g1->mass[INDEX(ix,iy,iz)];
	uint64_t * restrict id = g1->id[INDEX(ix,iy,iz)];

	size_t count2 = g2->counts[INDEX(ix,iy,iz)];	
	FLOAT * restrict x2 = g2->x[INDEX(ix,iy,iz)];
	FLOAT * restrict y2 = g2->y[INDEX(ix,iy,iz)];
	FLOAT * restrict z2 = g2->z[INDEX(ix,iy,iz)];

	/* now loop through each halo in this cell (i.e. each element of x1,y1,z1,label1) */
	for(size_t n = 0; n < count1; n++) {
	
	  uint64_t pcounts = 0;
	  _density_count_pairs(x1[n],y1[n],z1[n],\
			       x2,y2,z2,count2,	 \
			       &pcounts,rmax_sq,Lbox);

	  int iix,iiy,iiz;
	  for(iix=-1;iix<=1;iix++) {
	    for(iiy=-1;iiy<=1;iiy++) {
	      for(iiz=-1;iiz<=1;iiz++) {
		if(iix==0 && iiy==0 && iiz==0)
		  continue;

		int aix = (ix+iix+ngrid) % ngrid; // careful to ensure this is nonnegative!
		int aiy = (iy+iiy+ngrid) % ngrid;
		int aiz = (iz+iiz+ngrid) % ngrid;

		/* now count pairs with adjacent cells */
		size_t adj_count = g2->counts[INDEX(aix,aiy,aiz)];
		FLOAT * restrict adj_x = g2->x[INDEX(aix,aiy,aiz)];
		FLOAT * restrict adj_y = g2->y[INDEX(aix,aiy,aiz)];
		FLOAT * restrict adj_z = g2->z[INDEX(aix,aiy,aiz)];
		grid_id * restrict adj_label = g2->sample_excluded_from[INDEX(ix, iy, iz)];

		_density_count_pairs(x1[n],y1[n],z1[n],\
				     adj_x,adj_y,adj_z,adj_count,\
				     &pcounts,rmax_sq,Lbox);

	      }
	    }
	  }

	  /* add counts to global list */
	  halo_env_counts[q] = pcounts;
	  halo_id[q] = id[n];
	  halo_mass[q] = mass[n];
	  q++;
	  //	  printf("q: %ld\n", q);
	}
      }
    }
  }
}

#define SIMD_WIDTH 4

void _density_count_pairs(FLOAT x, FLOAT y, FLOAT z, FLOAT * restrict adj_x, FLOAT * restrict adj_y, FLOAT * restrict adj_z, size_t adj_count, uint64_t * pcounts, const double rmax_sq, const double Lbox)
{
	      /* // scalar version
	      size_t j;
	      for(j=0;j<adj_count;j++) {
	        double dist_sq = SQ(PERIODIC(x-adj_x[j])) + SQ(PERIODIC(y-adj_y[j])) + SQ(PERIODIC(z-adj_z[j]));
	        if(dist_sq < rmax_sq) {
	          pcounts++;
	        }
	      } */

  /* SIMD version */
  const size_t simd_size = adj_count/SIMD_WIDTH;
  size_t jj;
  for(jj=0;jj<simd_size;jj++)
    {
      double dist_sq[SIMD_WIDTH];
      size_t k;
#ifdef __INTEL_COMPILER
      __assume_aligned(adj_x, 32);
      __assume_aligned(adj_y, 32);
      __assume_aligned(adj_z, 32);
#endif
      //#pragma simd
      for(k=0;k<SIMD_WIDTH;k++)
	{
	  const size_t kk = k+jj*SIMD_WIDTH;
	  dist_sq[k] = SQ(PERIODIC(x-adj_x[kk])) + SQ(PERIODIC(y-adj_y[kk])) + SQ(PERIODIC(z-adj_z[kk]));
	}
      
      for(k=0;k<SIMD_WIDTH;k++) {
	if(!(dist_sq[k] > rmax_sq)) {
	  pcounts[0]++;
	}
      }
    }
  
  size_t k;
  for(k=((simd_size)*SIMD_WIDTH);k<adj_count;k++)
    {
      double dist_sq = SQ(PERIODIC(x-adj_x[k])) + SQ(PERIODIC(y-adj_y[k])) + SQ(PERIODIC(z-adj_z[k]));
      if(!(dist_sq > rmax_sq)) {
	pcounts[0]++;
      }
    }
}
