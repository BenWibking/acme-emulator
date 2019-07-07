#include "hash.h"

void cross_count_pairs_naive(FLOAT * x1, FLOAT * y1, FLOAT * z1, grid_id * label1, size_t npoints1, FLOAT * x2, FLOAT * y2, FLOAT * z2, grid_id * label2, size_t npoints2, uint64_t * pcounts, uint64_t * pcounts_jackknife, double *  bin_edges_sq, const int nbins, const int njack, const double Lbox)
{
  count_pairs_disjoint(x1,y1,z1,label1,x2,y2,z2,label2,npoints1,npoints2, \
		       pcounts,pcounts_jackknife,bin_edges_sq,nbins,njack,Lbox);
}

void cross_count_pairs(GHash * restrict g1, GHash * restrict g2, uint64_t * restrict pcounts, uint64_t * restrict pcounts_jackknife, double * restrict bin_edges_sq, int nbins, int njack)
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
  for(ix=0;ix<ngrid;ix++) {
    for(iy=0;iy<ngrid;iy++) {
      for(iz=0;iz<ngrid;iz++) {

	/* do this for each cell */
	size_t count1 = g1->counts[INDEX(ix,iy,iz)];
	size_t count2 = g2->counts[INDEX(ix,iy,iz)];
	FLOAT * restrict x1 = g1->x[INDEX(ix,iy,iz)];
	FLOAT * restrict y1 = g1->y[INDEX(ix,iy,iz)];
	FLOAT * restrict z1 = g1->z[INDEX(ix,iy,iz)];
	grid_id * restrict label1 = g1->sample_excluded_from[INDEX(ix, iy, iz)];
	
	FLOAT * restrict x2 = g2->x[INDEX(ix,iy,iz)];
	FLOAT * restrict y2 = g2->y[INDEX(ix,iy,iz)];
	FLOAT * restrict z2 = g2->z[INDEX(ix,iy,iz)];
	grid_id * restrict label2 = g2->sample_excluded_from[INDEX(ix, iy, iz)];
	
	count_pairs_disjoint(x1,y1,z1,label1,x2,y2,z2,label2,count1,count2, \
			     pcounts,pcounts_jackknife,bin_edges_sq,nbins,njack,Lbox);
	
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

	      count_pairs_disjoint(x1,y1,z1,label1,adj_x,adj_y,adj_z,adj_label,count1,adj_count, \
				   pcounts,pcounts_jackknife,bin_edges_sq,nbins,njack,Lbox);

	    }
	  }
	}
      }
    }
  }
}
