

#include "wavelet.h"


/*
 *	Perform the Haar Wavelet Transformation (HWT).
 */
wavelet_coeff_t* HWT( wavelet_data_t* wavelet_data_v, int k, int opt)
{
	if( opt == 1)
		return HWT_bottom_up( wavelet_data_v, k);
	else
		return HWT_basis_vector( wavelet_data_v, k);
}

/*
 *	Perform HWT with a bottom-up approach.
 *
 *	Issue 1: should the scaling factor be used in our case?
 *		-- Not necessary, but for the purpose of possible future use, 
 *		the scalling factor is used in this implementation.
 *
 *	Issue 2: how to maintain the top-k?
 *		-- 
 */
wavelet_coeff_t* HWT_bottom_up( wavelet_data_t* wavelet_data_v, int k)
{
	int level, i, idx, n, m;
	LOC_TYPE scale_factor;

	wavelet_coeff_t* wavelet_coeff_v;

	//Complete binary tree;
	LOC_TYPE* aver_coeff;
	LOC_TYPE* diff_coeff;

	n = wavelet_data_v->n;
	m = wavelet_data_v->m;

	//
	aver_coeff = ( LOC_TYPE*)malloc( 2 * n * sizeof( LOC_TYPE));
	memset( aver_coeff, 0, 2 * n * sizeof( LOC_TYPE));

	diff_coeff = ( LOC_TYPE*)malloc( n * sizeof( LOC_TYPE));
	memset( diff_coeff, 0, n * sizeof( LOC_TYPE));

	//s
	emp_stat_v.memory_v += 2 * n * sizeof( LOC_TYPE) + n * sizeof( LOC_TYPE);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;

	//
	for( i=n; i<2*n; i++)
		aver_coeff[ i] = wavelet_data_v->value[ i-n+1];

	//level by level;
	//the index of the first element 2^level.
	idx = n;
	for( level=m-1; level>=0; level--)
	{
		i = idx / 2;					//Note that i = 2^level
		scale_factor = (LOC_TYPE) sqrt( n / i);	//
		for( ; i<idx; i++)
		{
			aver_coeff[ i] = ( aver_coeff[ 2*i+1] + aver_coeff[ 2*i]) / 2;
			aver_coeff[ i] *= scale_factor;
			
			diff_coeff[ i] = ( aver_coeff[ 2*i+1] - aver_coeff[ 2*i]) / 2;
			diff_coeff[ i] *= scale_factor;
		}

		//
		idx /= 2;
	}

	//
	/*
	wavelet_coeff_v->coeff[ 1] = aver_coeff[ 1];
	for( i=2; i<=n; i++)
	{
		wavelet_coeff_v->coeff[ i] = diff_coeff[ i-1];
	}
	*/
	
	//Maintain the coefficients;
	diff_coeff[ 0] = aver_coeff[ 1];

	//Get the top-k coefficients;
	wavelet_coeff_v = get_top_k( diff_coeff, n, k);

	//Release the resources.
	free( aver_coeff);
	free( diff_coeff);

	//S
	emp_stat_v.memory_v -= 2 * n * sizeof( LOC_TYPE) + n * sizeof( LOC_TYPE);
	//S

	return wavelet_coeff_v;
}

/*
 *	Perform HWT with a basis vector approach.
 */
wavelet_coeff_t* HWT_basis_vector( wavelet_data_t* wavelet_data_v, int k)
{

	return NULL;
}

/*
 *	Get the top-k values in @values and store then in @wavelet_coeff_v;
 *		-- the implementation is based on a heap structure.
 */
wavelet_coeff_t* get_top_k( LOC_TYPE* values, int size, int k)
{	
	int i, top;
	
	b_heap_t* b_h;
	wavelet_coeff_t* wavelet_coeff_v;
	
	MIN_HEAP_OPT = 0;	//indicating that a max-heap is used.

	//
	wavelet_coeff_v = wavelet_coeff_alloc( k, size);
	
	//Initialize the heap;
	b_h = alloc_b_heap( size);

	//Construct the heap;
	for( i=0; i<size; i++)
	{
		b_h->obj_arr[ i+1].key = values[ i];

		insert_b_heap( b_h, i+1);
	}

	//Get the top-k;
	for( i=1; i<=k; i++)
	{
		top = get_top( b_h);

		wavelet_coeff_v->coeff[ i] = b_h->obj_arr[ top].key;
		wavelet_coeff_v->idx[ i] = top - 1;		
	}

	//Release the resources;
	release_b_heap( b_h);

	return wavelet_coeff_v;
}

/*
 * Perform the Haar Wavelet Detransformation.
 *
 *  1. How to fit back the top-k coefficients in the hierarchical tree structure?
 *	-- the index for each coefficient should also be maintained.
 *
 *	2. A top-down approach is adopted.
 *	-- reverse the way back.
 */
wavelet_data_t* HWDT( wavelet_coeff_t* wavelet_coeff_v)
{
	int i, k, n, m, level, idx;
	LOC_TYPE scale_factor;
	LOC_TYPE* aver_coeff, *diff_coeff;

	wavelet_data_t* wavelet_data_v;

	k = wavelet_coeff_v->k;
	n = wavelet_coeff_v->n;
	m = log2( n);

	//
	wavelet_data_v = wavelet_data_alloc( n);

	//
	aver_coeff = ( LOC_TYPE*)malloc( 2 * n * sizeof( LOC_TYPE));
	memset( aver_coeff, 0, 2 * n * sizeof( LOC_TYPE));

	diff_coeff = ( LOC_TYPE*)malloc( n * sizeof( LOC_TYPE));
	memset( diff_coeff, 0, n * sizeof( LOC_TYPE));

	//s
	emp_stat_v.memory_v += 2 * n * sizeof( LOC_TYPE) + n * sizeof( LOC_TYPE);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	//Restore the differencing coefficients;
	for( i=1; i<=k; i++)
	{
		idx = wavelet_coeff_v->idx[ i];
		
		diff_coeff[ idx] = wavelet_coeff_v->coeff[ i];
	}
	aver_coeff[ 1] = diff_coeff[ 0];
		
	//Recover the original values;
	//level by level
	idx = 1;
	scale_factor = ( LOC_TYPE)sqrt( n / idx);
	aver_coeff[ 1] /= scale_factor;
	diff_coeff[ 1] /= scale_factor;

	for( level=1; level<=m; level++)
	{
		idx *= 2;
		scale_factor = ( LOC_TYPE)sqrt( n / idx);
		
		for( i=idx; i<2*idx; i+=2)
		{	
			//
			aver_coeff[ i] = aver_coeff[ i/2] - diff_coeff[ i/2];
			aver_coeff[ i+1] = aver_coeff[ i/2] + diff_coeff[ i/2];
	
			if( level != m)
			{
				//scale_factor;
				aver_coeff[ i] /= scale_factor;
				aver_coeff[ i+1] /= scale_factor;
				diff_coeff[ i] /= scale_factor;
				diff_coeff[ i+1] /= scale_factor;
			}
		}
	}

	//
	for( i=1; i<=n; i++)
		wavelet_data_v->value[ i] = aver_coeff[ i+n-1];

	//Release the resources;
	free( aver_coeff);
	free( diff_coeff);

	//s
	emp_stat_v.memory_v -= 2 * n * sizeof( LOC_TYPE) + n * sizeof( LOC_TYPE);
	//s

	return wavelet_data_v;
}



/*
 *	Test the APIs for HWT;
 */
void test_HWT( )
{
	wavelet_config_t* wavelet_config_v;
	wavelet_data_t* wavelet_data_v1, *wavelet_data_v2;
	wavelet_coeff_t* wavelet_coeff_v;

	//
	wavelet_config_v = read_wavelet_config( );

	//
	wavelet_data_v1 = read_wavelet_data( wavelet_config_v);

	//
	wavelet_coeff_v = HWT( wavelet_data_v1, wavelet_config_v->k, 1);

	//
	print_wavelet_coeff( wavelet_coeff_v, wavelet_config_v->coeff_f_name_out);

	//
	wavelet_data_v2 = HWDT( wavelet_coeff_v);

	//
	print_wavelet_data( wavelet_data_v2, wavelet_config_v->f_name_out);

	//Release the resource;
	free( wavelet_config_v);
	wavelet_data_release( wavelet_data_v1);
	wavelet_data_release( wavelet_data_v2);
	wavelet_coeff_release( wavelet_coeff_v);
}

wavelet_config_t* read_wavelet_config( )
{
	FILE* c_fp;
	wavelet_config_t* wavelet_config_v;

	//
	wavelet_config_v = ( wavelet_config_t*)malloc( sizeof( wavelet_config_t));
	memset( wavelet_config_v, 0, sizeof( wavelet_config_t));
	
	if( ( c_fp = fopen( WAVELET_CONFIG_FILE, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open %s [read_wavelet_config].\n", WAVELET_CONFIG_FILE);
		exit( 0);
	}

	//
	fscanf( c_fp, "%i%i%s%s%s", &wavelet_config_v->n, &wavelet_config_v->k,
								wavelet_config_v->f_name_in, wavelet_config_v->f_name_out,
								wavelet_config_v->coeff_f_name_out);

	return wavelet_config_v;
}

wavelet_data_t* read_wavelet_data( wavelet_config_t* config_v)
{
	int i;
	FILE* i_fp;

	wavelet_data_t* wavelet_data_v;

	wavelet_data_v = wavelet_data_alloc( config_v->n);

	if( ( i_fp = fopen( config_v->f_name_in, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open %s [read_data_wavelet].\n");
		exit( 0);
	}
	
	for( i=0; i<config_v->n; i++)
		fscanf( i_fp, "%f", wavelet_data_v->value + i + 1);

	//Release the resources;
	fclose( i_fp);

	return wavelet_data_v;
}

void print_array( LOC_TYPE* array, int sta, int end, char* f_name)
{
	int i;
	FILE* o_fp;

	if( ( o_fp = fopen( f_name, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open %s [print_data_wavelet].\n");
		exit( 0);
	}

	for( i=sta; i<=end; i++)
		fprintf( stdout, "%f\n", array[ i]);
	fprintf( stdout, "\n");
	
	//Release the resources;
	fclose( o_fp);

	return;

}

void print_wavelet_data( wavelet_data_t* wavelet_data_v, char* f_name)
{
	print_array( wavelet_data_v->value, 1, wavelet_data_v->n, f_name);
}

void print_wavelet_coeff( wavelet_coeff_t* wavelet_coeff_v, char* f_name)
{
	print_array( wavelet_coeff_v->coeff, 1, wavelet_coeff_v->k, f_name);	
}
