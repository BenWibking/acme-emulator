#include "hash1d.h"

MHash* allocate_1d_hash(int nbins, double * bin_edges, size_t npoints, halo_metadata * h)
{
  if ((nbins <= 0) || (npoints <= 0)) {
    return (void*)0;
  }

  MHash * m = my_malloc(sizeof(MHash));
  m->nbins = nbins;
  m->bin_edges = bin_edges;
  m->counts = my_malloc(nbins*sizeof(size_t));
  m->allocated = my_malloc(nbins*sizeof(size_t));
  m->h = my_malloc(nbins*sizeof(halo_metadata*));

  int i;
  for(i=0;i<nbins;i++)
    {
      m->counts[i] = 0;
      m->allocated[i] = 0;
    }

  /* compute needed allocation sizes for each bin */
  size_t n;
  for(n=0;n<npoints;n++) {
    double s = h[n].mass;
    /* linear search */
    size_t j;
    int alloc_flag = 0;
    for(j=0;j<nbins;j++) {
      if((s < bin_edges[j+1]) && (s >= bin_edges[j])) {
	/* add space in bin[j] */
	m->allocated[j]++;
	alloc_flag = 1;
	break; /* need to make sure each halo is placed in a bin!! */
      }
    }
    if (alloc_flag == 0) {
      /* must put particle in either underflow or overflow bin */
      if(s < bin_edges[0]) { /* underflow */
	m->allocated[0]++;
      } else if (s > bin_edges[nbins]) { /* overflow */
	m->allocated[nbins-1]++;
      } else { /* something is wrong */
	printf("WARNING: a particle is not allocated to a bin!\n");
      }
    }
  }

  /* allocate each bin */
  size_t j;
  for(j=0;j<nbins;j++) {
    size_t count = m->allocated[j];
    m->h[j] = my_malloc(count*sizeof(halo_metadata));
  }

  /* add elements to each bin */
  for(n=0;n<npoints;n++) {
    double s = h[n].mass;
    /* linear search */
    size_t j;
    int alloc_flag = 0;
    for(j=0;j<nbins;j++) {
      if((s < bin_edges[j+1]) && (s >= bin_edges[j])) {
	size_t count = m->counts[j];
	m->h[j][count] = h[n]; /* or memcpy? */
	(m->counts[j])++;
	alloc_flag = 1;
	break; /* need to make sure each is placed in a bin!! */
      }
    }
    if (alloc_flag == 0) {
      /* must put particle in either underflow or overflow bin */
      if(s < bin_edges[0]) { /* underflow */
	size_t count = m->counts[0];
	m->h[0][count] = h[n];
	m->counts[0]++;
      } else if (s > bin_edges[nbins]) { /* overflow */
	size_t count = m->counts[nbins-1];
	m->h[nbins-1][count] = h[n];
	m->counts[nbins-1]++;
      }
    }
  }
  
  return m;
}

int compare_halo_metadata_by_density(const void* a, const void* b)
{
  halo_metadata * haloA = (halo_metadata*)a;
  halo_metadata * haloB = (halo_metadata*)b;

  if( haloA->density < haloB->density ) return -1;
  if( haloA->density > haloB->density ) return 1;
  return 0;
}

int compare_halo_metadata_by_id(const void* a, const void* b)
{
  halo_metadata * haloA = (halo_metadata*)a;
  halo_metadata * haloB = (halo_metadata*)b;

  if( haloA->id < haloB->id ) return -1;
  if( haloA->id > haloB->id ) return 1;
  return 0;
}

void sort_1d_hash(MHash * m)
{
  /* sort each bin within the 1d hash */
  size_t nbins = m->nbins;

  size_t j;
  for(j=0;j<nbins;j++) {
    /* sort halos within this bin */
    size_t array_size = m->counts[j];
    halo_metadata * array_to_sort = m->h[j];

    //    printf("mass bin: %lld array_size: %lld\n", j, array_size);
    
    /* use standard C qsort with custom sorter function */
    qsort(array_to_sort, array_size, sizeof(halo_metadata), compare_halo_metadata_by_density);

    /* walk through array_to_sort in order, adding percentiles */
    size_t i;
    for(i=0;i<array_size;i++) {
      float percentile = ((float)i)/((float)(array_size));
      array_to_sort[i].percentile = percentile;
      //      printf("i: %lld mass: %f percentile: %f\n", i, array_to_sort[i].mass, array_to_sort[i].percentile);
    }
  }
}

void linearize_1d_hash(MHash * m, size_t len, halo_metadata * linear_halos)
{
  size_t nbins = m->nbins;
  size_t j;
  size_t q = 0;
  for(j=0;j<nbins;j++) {
    size_t array_size = m->counts[j];
    halo_metadata * this_halos = m->h[j];
    size_t i;
    for(i=0;i<array_size;i++) {
      linear_halos[q] = this_halos[i];
      q++;
    }
  }
}
