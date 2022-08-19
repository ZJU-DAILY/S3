/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



#include "min_error.h"

//Starting from 17 July, 2013.
//For the Min-Error problem.
error_space_t* error_space_alloc( int pos_n)
{
	int i;
	error_space_t* error_space_v;

	error_space_v = ( error_space_t*)malloc( sizeof( error_space_t));
	memset( error_space_v, 0, sizeof( error_space_t));

	error_space_v->pos_n = pos_n;

	error_space_v->matrix = ( float**)malloc( ( pos_n + 1) * sizeof( float*));
	memset( error_space_v->matrix, 0, ( pos_n + 1) * sizeof( float*));

	for( i=0; i<pos_n+1; i++)
	{
		error_space_v->matrix[ i] = ( float*)malloc( ( pos_n + 1) * sizeof( float));
		memset( error_space_v->matrix[ i], 0, ( pos_n + 1) * sizeof( float));
	}

	//s
	emp_stat_v.memory_v += sizeof( error_space_t) + ( pos_n + 1) * sizeof( float*) +
							( pos_n + 1) * ( pos_n + 1) * sizeof( float);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	return error_space_v;
}

void error_space_release( error_space_t* error_space_v)
{
	int i;

	for( i=0; i<error_space_v->pos_n+1; i++)
		free( error_space_v->matrix[ i]);

	//t
	int pos_n;
	
	pos_n = error_space_v->pos_n;
	emp_stat_v.memory_v -= sizeof( error_space_t) + ( pos_n + 1) * sizeof( float*) +
							( pos_n + 1) * ( pos_n + 1) * sizeof( float);
	//t

	free( error_space_v);

	return;
}

/*
 *	Construct the error space using the straightforward method,
 *	which has its cost of O(n^3).
 */
error_space_t* const_error_space_v1( R_list_t* R_list_v)
{
	int i, j, pos_n;

	error_space_t* error_space_v;
	R_node_t* R_node_v1, *R_node_v2;

	pos_n = R_list_v->size + 1;
	error_space_v = error_space_alloc( pos_n);

	if( pos_n <= 2)
		return error_space_v;

	i = 1;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1->next != NULL)
	{
		j = i+2;
		R_node_v2 = R_node_v1->next;

		while( R_node_v2 != NULL)
		{
			error_space_v->matrix[ i][ j] = calc_error( R_node_v1, R_node_v2);

			j++;
			R_node_v2 = R_node_v2->next;
		}

		i++;
		R_node_v1 = R_node_v1->next;
	}

	return error_space_v;
}

/*
 *	Find @slope_v's closest direction among all those maintained in a binary search tree @bst_v.
 */
float find_closest_slope( bst_t* bst_v, float slope_v)
{
	int child_loc;
	float slope_v1, slope_v2, diff_v1, diff_v2;
	bst_node_t* pre, *suc, *x, *y, *z;

	//Search for the appropriate location for the key.
	child_loc = 0;
	y = NULL;
	x = bst_v->root;
	while( x != NULL)
	{
		y = x;

		if( slope_v < x->key)
		{
			x = x->left;
			child_loc = -1;
		}
		else
		{
			x = x->right;
			child_loc = 1;
		}
	}

	//Search for the successor.
	//There is a bug here.
	z = x;
	suc = y;
	while( suc != NULL && z == suc->right && child_loc == 1)
	{
		z = suc;
		suc = suc->p;
	}

	//Search for the predecessor.
	z = x;
	pre = y;
	while( pre != NULL && z == pre->left && child_loc == -1)
	{
		z = pre;
		pre = pre->p;
	}	

	//t
	if( pre == NULL && suc == NULL)
	{
		fprintf( stderr, "pre and suc cannot be NULL simultaneously [find_closest_slope].\n");
		exit( 0);
	}
	//t

	//
	if( pre == NULL)
		slope_v1 = bst_v->max;
	else
		slope_v1 = pre->key;

	if( suc == NULL)
		slope_v2 = bst_v->min;
	else
		slope_v2 = suc->key;

	//
	diff_v1 = calc_angular_diff( slope_v, slope_v1);
	diff_v2 = calc_angular_diff( slope_v, slope_v2);

	return diff_v1 < diff_v2 ? slope_v1 : slope_v2;
}

/*
 *	Calculate the maximum angular difference between @slope_v and those maintained in a binary search tree @bst_v.
 */
float calc_max_angular_diff( bst_t* bst_v, float slope_v)
{
	float slope_v1, slope_v2;

	//
	slope_v1 = modulo_op( slope_v + PI, 2 * PI);

	//
	slope_v2 = find_closest_slope( bst_v, slope_v1);

	return calc_angular_diff( slope_v, slope_v2);
}

/*
 *	Construct the error space using an advanced method,
 *	which has its cost of O(n^2 log n).
 */
error_space_t* const_error_space_v2( R_list_t* R_list_v)
{
	int i, j, pos_n;
	float slope_v;

	error_space_t* error_space_v;
	R_node_t* R_node_v1, *R_node_v2;

	bst_t* bst_v;
	bst_node_t* bst_node_v;

	pos_n = R_list_v->size + 1;
	error_space_v = error_space_alloc( pos_n);

	if( pos_n <= 2)
		return error_space_v;


	i = 1;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1->next != NULL)
	{
		//s
		if( R_node_v1->id1 == 71)
			printf( "");
		//s

		//
		bst_v = bst_ini( );
		bst_node_v = bst_insert_key( bst_v, R_node_v1->slope);

		j = i+2;
		R_node_v2 = R_node_v1->next;

		while( R_node_v2 != NULL)
		{
			//
			bst_insert_key( bst_v, R_node_v2->slope);

			//
			slope_v = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);


			//error_space_v->matrix[ i][ j] = calc_error( R_node_v1, R_node_v2);
			error_space_v->matrix[ i][ j] = calc_max_angular_diff( bst_v, slope_v);

			j++;
			R_node_v2 = R_node_v2->next;
		}

		i++;
		R_node_v1 = R_node_v1->next;

		//update the binary search tree.
		bst_release( bst_v);
	}

	return error_space_v;
}

void print_error_space( error_space_t* error_space_v)
{
	int i, j;

	for( i=1; i<error_space_v->pos_n; i++)
	{
		//for( j=1; j<=i; j++)
		//	printf( "0  ");
		for( j=i+1; j<=error_space_v->pos_n; j++)
			printf( "%f  ", error_space_v->matrix[ i][ j]);

		printf( "\n");
	}
}

/*
 *	The DP algorithm for the Min-Error problem.
 */
float DP_Error( R_list_t* R_list_v, int W, int ver_opt)
{
	int i, k, h, pos_n;
	float min_error, error_v, rtn;
	float** E;

	error_space_t* error_space_v;

	pos_n = R_list_v->size + 1;


	if( pos_n <= 1)
		return 0;
	if( W <= 1)
		return PI;

	//Initialize the solution space.
	E = ( float**)malloc( pos_n * sizeof( float*));
	memset( E, 0, pos_n * sizeof( float*));

	for( i=0; i<pos_n; i++)
	{
		E[ i] = ( float*)malloc( ( W + 1) * sizeof( float));
		memset( E[ i], 0, ( W + 1) * sizeof( float));
	}

	//s
	emp_stat_v.memory_v += pos_n * sizeof( float*) + pos_n * ( W + 1) * sizeof( float);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s
		
	//Compute the error space.
	if( ver_opt == 1)
		error_space_v = const_error_space_v1( R_list_v);
	else
		error_space_v = const_error_space_v2( R_list_v);

	//t
	//print_error_space( error_space_v);
	//t

	//The DP process.
	for( i=1; i<=pos_n-1; i++)
	{
		E[ i][ 2] = error_space_v->matrix[ i][ pos_n];
	}

	for( k=3; k<=W; k++)
	{
		for( i=1; i<=pos_n-k; i++)
		{
			//E[ i][ k] = 
			min_error = FLT_MAX;

			for( h=i+1; h<=pos_n-1; h++)
			{
				error_v = error_space_v->matrix[ i][ h] > E[ h][ k-1] ?
					error_space_v->matrix[ i][ h] : E[ h][ k-1];

				if( error_v < min_error)
					min_error = error_v;
			}

			E[ i][ k] = min_error;
		}
	}

	rtn = E[ 1][ W];

	//Release the resources.
	for( i=0; i<pos_n; i++)
		free( E[ i]);
	free( E);
	
	error_space_release( error_space_v);

	//s
	emp_stat_v.memory_v -= pos_n * sizeof( float*) + pos_n * ( W + 1) * sizeof( float);
	//s

	return rtn;
}

/*
 *	Form the 2D matrix to 1D array.
 */
float* reform_error_space( error_space_t* error_space_v)
{
	int i, j, cnt, size;
	float* error_array_v;

	size = error_space_v->pos_n * ( error_space_v->pos_n - 1) / 2;

	error_array_v = ( float*)malloc( size * sizeof( float));
	memset( error_array_v, 0, size * sizeof( float));

	//s
	emp_stat_v.memory_v += size * sizeof( float);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	cnt = 0;
	for( i=1; i<error_space_v->pos_n; i++)
	{
		for( j=i+1; j<=error_space_v->pos_n; j++)
		{
			error_array_v[ cnt++] = error_space_v->matrix[ i][ j];
		}
	}

	return error_array_v;
}

/*
 *
 */
id_list_t* Error_Search( R_list_t* R_list_v, int W)
{
	int pos_n, size, left, right, mid;
	//float min_affordable_error;
	float* error_array_v;

	error_space_t* error_space_v;
	id_list_t* id_list_v, *id_list_v_min;

	//
	pos_n = R_list_v->size + 1;
	size = pos_n * ( pos_n - 1) / 2;

	//
	error_space_v = const_error_space_v2( R_list_v);

	error_array_v = reform_error_space( error_space_v);

	qsort( error_array_v, size, sizeof( float), compare);

	//Binary search;
	//min_affordable_error = FLT_MAX;
	id_list_v_min = NULL;
	left = 0;
	right = size - 1;
	while( left < right)
	{
		mid = ( left + right) / 2;

		id_list_v = SP_Method_prac_and_theo( R_list_v, error_array_v[ mid]);

		if( id_list_v->pos_id_n <= W)
		{
			right = mid;
	
			id_list_release( id_list_v_min);

			id_list_v_min = id_list_v;

			id_list_v = NULL;
		}
		else
			left = mid + 1;
		
		id_list_release( id_list_v);
	}
	
	//min_affordable_error = error_array_v[ right];

	//Release the resources.
	error_space_release( error_space_v);
	free( error_array_v);

	//s
	emp_stat_v.memory_v -= size * sizeof( float);
	//s

	//return min_affordable_error;
	return id_list_v_min;
}

span_space_t* span_space_alloc( int pos_n)
{
	span_space_t* span_space_v;

	span_space_v = ( span_space_t*)malloc( sizeof( span_space_t));
	memset( span_space_v, 0, sizeof( span_space_t));

	span_space_v->head = ( array_node_t*)malloc( sizeof( array_node_t));
	memset( span_space_v->head, 0, sizeof( array_node_t));

	span_space_v->slope_array = ( float*)malloc( pos_n * sizeof( float));
	memset( span_space_v->slope_array, 0, pos_n * sizeof( float));

	span_space_v->pos_n = pos_n;

	//s
	emp_stat_v.memory_v += sizeof( span_space_t) + sizeof( array_node_t) + pos_n * sizeof( float);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s
	
	return span_space_v;
}

void print_span_space( span_space_t* span_space_v)
{
	//
}

void span_space_release( span_space_t* span_space_v)
{
	array_node_t* iter_1, *iter_2;

	iter_1 = span_space_v->head;
	while( iter_1 != NULL)
	{
		iter_2 = iter_1->next;

		free( iter_1);

		//s
		emp_stat_v.memory_v -= sizeof( array_node_t);
		//s

		iter_1 = iter_2;
	}

	free( span_space_v->slope_array);
	
	//s
	emp_stat_v.memory_v -= span_space_v->pos_n * sizeof( float);
	//s

	free( span_space_v);

	//s
	emp_stat_v.memory_v -= sizeof( span_space_t);
	//s
}

/*
 *	Add an array_node to the span_sapce.
 */
void add_span_space_entry( span_space_t* span_space_v, int s, int e, int j)
{
	array_node_t* array_node_v;

	array_node_v = ( array_node_t*)malloc( sizeof( array_node_t));
	memset( array_node_v, 0, sizeof( array_node_t));

	//s
	emp_stat_v.memory_v += sizeof( array_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s
	
	array_node_v->array_v.s = s;
	array_node_v->array_v.e = e;
	array_node_v->array_v.j = j;

	array_node_v->next = span_space_v->head->next;
	array_node_v->pre = span_space_v->head;

	span_space_v->head->next = array_node_v;
	if( array_node_v->next != NULL)
		array_node_v->next->pre = array_node_v;

	span_space_v->array_n ++;
	span_space_v->size += e - s + 1;

	return;
}

/*
 *
 */
span_space_t* const_span_space( R_list_t* R_list_v)
{
	int j, pos_n, cnt;
	
	R_node_t* R_node_v;
	span_space_t* span_space_v;

	pos_n = R_list_v->size + 1;

	span_space_v = span_space_alloc( pos_n);

	if( pos_n <= 1)
		return span_space_v;

	//compute the set of directions.
	cnt = 1;
	R_node_v = R_list_v->head->next;
	while( R_node_v != NULL)
	{
		span_space_v->slope_array[ cnt++] = R_node_v->slope;
		
		R_node_v = R_node_v->next;
	}
	
	//sort the set of directions.
	qsort( span_space_v->slope_array + 1, pos_n - 1, sizeof( float), compare);

	//maintain the arrays.
	for( j=1; j<pos_n-1; j++)
	{
		add_span_space_entry( span_space_v, 1, j, j);

		add_span_space_entry( span_space_v, j+1, pos_n-1, j);
	}

	add_span_space_entry( span_space_v, 1, j, j);

	return span_space_v;
}

/*
 *	Get the median of the array in @array_node_v.
 */
float get_median( span_space_t* span_space_v, array_node_t* array_node_v)
{
	int s, e, j;

	s = array_node_v->array_v.s;
	e = array_node_v->array_v.e;
	j = array_node_v->array_v.j;
	
	return get_matrix_entry( span_space_v, ( s + e) / 2, j);
}

/*
 *	Check whether span_v is a pivot.
 *
 *	@return
 *		= -1: 
 *		= 0:	 pivot.
 *		= 1:
 */
int	is_a_pivot( span_space_t* span_space_v, float span_v)
{
#ifndef WIN32
	long long cnt_v1, cnt_v2, cnt_v3, size;
#else
	int cnt_v1, cnt_v2, cnt_v3, size;
#endif

	float median, diff;

	array_node_t* array_node_v;

	cnt_v1 = 0;
	cnt_v2 = 0;
	cnt_v3 = 0;
	array_node_v = span_space_v->head->next;
	while( array_node_v != NULL)
	{
		median = get_median( span_space_v, array_node_v);
		size = array_node_v->array_v.e - array_node_v->array_v.s + 1;

		diff = median - span_v;

		//if( median < span_v)
		if( diff < -1e-6)
			cnt_v1 += size;
		else if( diff > 1e-6)
			cnt_v3 += size;
		//else if( abs_f( median - span_v) >= -1e-6 && abs_f( median - span_v) <= 1e-6)
		//else if( abs_f( median - span_v) >= -1e-6 && abs_f( median - span_v) <= 1e-6)
		else
			cnt_v2 += size;

		array_node_v = array_node_v->next;
	}
		
	if( cnt_v1 + cnt_v2 >= span_space_v->size / 2 &&
		cnt_v3 + cnt_v2 >= span_space_v->size / 2)
		return 0;
	else if( cnt_v1 + cnt_v2 < span_space_v->size / 2)
		return -1;
	else
		return 1;
}

//pivot finding.
/*
 *
 */
float find_pivot( span_space_t* span_space_v)
{
	int cnt, left, right, mid, res;
	float pivot;
	float* medians;

	array_node_t* array_node_v;

	//compute the set of the medians of the arrays.
	medians = ( float*)malloc( span_space_v->array_n * sizeof( float));
	memset( medians, 0, span_space_v->array_n * sizeof( float));

	//s
	emp_stat_v.memory_v += span_space_v->array_n * sizeof( float);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	cnt = 0;
	array_node_v = span_space_v->head->next;
	while( array_node_v != NULL)
	{
		medians[ cnt++] = get_median( span_space_v, array_node_v);

		array_node_v = array_node_v->next;
	}

	//sort the set of the medians of the arrays.
	qsort( medians, span_space_v->array_n, sizeof( float), compare);

	//binary search.
	left = 0;
	right = span_space_v->array_n - 1;
	while( left <= right)
	{
		mid = ( left + right) / 2;

		//pivot condition checking.
		res = is_a_pivot( span_space_v, medians[ mid]);
		
		if( res == 0)
		{
			//medians[ mid] is pivot.
			break;
		}
		else if( res < 0)
		{
			left = mid + 1;
		}
		else	//res > 0
		{
			right = mid - 1;
		}
	}

	//t
	if( left > right)
	{
		fprintf( stderr, "No pivot [find_pivot].\n");
		exit( 0);
	}
	//t

	pivot = medians[ mid];

	//Release the resources.
	free( medians);

	//s
	emp_stat_v.memory_v -= span_space_v->array_n * sizeof( float);
	//s

	return pivot;
}

//span affordability check.

/*
 *	The implementation of "span_affordability_check".
 *
 *	1. the binary search tree maintains the sorted list of the directions.
 *	2. the heap maintains the angular range with the greatest span.
 */
id_list_t* span_affordability_check( R_list_t* R_list_v, float span_v)
{
	int i, pos_n, h_cnt;

	bst_t* bst_v;
	bst_node_t* bst_node_v, *pre, *suc;
	b_heap_t* b_heap_v;
	R_node_t* R_node_v1, *R_node_v2;
	
	id_list_t* id_list_v;

	id_list_v = id_list_alloc( );

	pos_n = R_list_v->size + 1;

	if( pos_n == 1)
	{
		fprintf( stderr, "Trival trajectory with one postions[span_affordability_check].\n");
		exit( 0);
	}

	//The starting position.
	i = 1;
	R_node_v1 = R_list_v->head->next;	//R_list_v1->id1 indicates i.
	R_node_v2 = R_node_v1;				//R_list_v2->id2 indicates j, currently j is equal to i+1.
	
	append_id_list_entry( id_list_v, R_node_v1->id1);

	while( R_node_v2 != NULL)
	{
		//This indicates that j <= pos_n;

		bst_v = bst_ini( );
		bst_node_v = bst_insert_key( bst_v, R_node_v2->slope);
		
		b_heap_v = alloc_b_heap( pos_n - 1);
		h_cnt = 1;
		
		b_heap_v->obj_arr[ h_cnt].key = 2 * PI;
		insert_b_heap( b_heap_v, h_cnt);
		bst_node_v->range_inx = h_cnt;

		h_cnt ++;

		//mcar( theat[i: j]) = 2 * PI - the maximum span of the angular ranges.
		while( R_node_v2 != NULL && ( 2 * PI - b_heap_v->obj_arr[ b_heap_v->h_arr[ 1]].key) <= span_v)
		{
			R_node_v2 = R_node_v2->next;

			if( R_node_v2 == NULL)
				break;

			//
			bst_node_v = bst_insert_key( bst_v, R_node_v2->slope);

			//Update the the corresponding ranges.
			pre = bst_predecessor( bst_node_v);
			suc = bst_successor( bst_node_v);

			if( pre == NULL)
			{
				pre = bst_max( bst_v->root);
			}
			if( suc == NULL)
			{
				suc = bst_min( bst_v->root);
			}

			//
			b_heap_v->obj_arr[ pre->range_inx].key = modulo_op( bst_node_v->key - pre->key, 2 * PI);
			update_key( b_heap_v, pre->range_inx);

			b_heap_v->obj_arr[ h_cnt].key = modulo_op( suc->key - bst_node_v->key, 2 * PI);
			insert_b_heap( b_heap_v, h_cnt);
			bst_node_v->range_inx = h_cnt;

			h_cnt ++;
		}
		
		if( R_node_v2 == NULL)
		{
			append_id_list_entry( id_list_v, R_list_v->rear->id2);
			break;
		}
		else
			append_id_list_entry( id_list_v, R_node_v2->id1);

		R_node_v1 = R_node_v2;

		//
		bst_release( bst_v);
		release_b_heap( b_heap_v);
	}

	//Release the resources.
	bst_release( bst_v);
	release_b_heap( b_heap_v);

	return id_list_v;
}

/*
 *	Get \theta[ i][ j].
 */
float get_matrix_entry( span_space_t* span_space_v, int i, int j)
{
	if( j >= i)
		return span_space_v->slope_array[ j] - span_space_v->slope_array[ i];
	else
		return 2 * PI - ( span_space_v->slope_array[ i] - span_space_v->slope_array[ j]);
}

/*
 *	tag 
 *		= -1: 
 *		=  1: 
 */
void prune_span_space( span_space_t* span_space_v, float pivot, int tag)
{
	int s, e, j;
	float median, diff;

	array_node_t* array_node_v, *array_node_v_tmp;

	array_node_v = span_space_v->head->next;
	while( array_node_v != NULL)
	{
		s = array_node_v->array_v.s;
		e = array_node_v->array_v.e;
		j = array_node_v->array_v.j;
		
		median = get_matrix_entry( span_space_v, ( s + e) / 2, j);

		diff = median - pivot;
		
		if( tag == -1 && diff >= -1e-6)//
		{
			//Prune "at least".
			span_space_v->size -= ( s + e) / 2  - s + 1;

			array_node_v->array_v.s = ( s + e ) / 2 + 1;
		}
		else if( tag == 1 && diff <= 1e-6)//median <= pivot)
		{	
			//Prune "at most".
			span_space_v->size -= e - ( s + e) / 2 + 1;

			array_node_v->array_v.e = ( s + e) / 2 - 1;
		}
		
		if( array_node_v->array_v.s > array_node_v->array_v.e)
		{
			//delete the array.
			array_node_v->pre->next = array_node_v->next;
			if( array_node_v->next != NULL)
				array_node_v->next->pre = array_node_v->pre;
			
			array_node_v_tmp = array_node_v;
			array_node_v = array_node_v->pre;
			
			free( array_node_v_tmp);

			//s
			emp_stat_v.memory_v -= sizeof( array_node_t);
			//s

			span_space_v->array_n --;
		}

		array_node_v = array_node_v->next;
	}

	return;
}

/*
 * Compute the error of a simplified trajectory indicated by @id_list_v.
 */
float comp_simp_error( R_list_t* R_list_v, id_list_t* id_list_v)
{
	float max_diff, error_v;

	R_node_t* R_node_v1, *R_node_v2;
	id_node_t* id_node_v;

	if( id_list_v->pos_id_n < 2)
	{
		fprintf( stderr, "Invalid simplified trajectory [comp_simp_error].\n");
		exit( 0);
	}

	error_v = 0;
	id_node_v = id_list_v->head->next;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1 != NULL)
	{
		while( R_node_v1 != NULL && R_node_v1->id1 != id_node_v->pos_id)
			R_node_v1 = R_node_v1->next;

		//t
		if( R_node_v1 == NULL)
		{
			fprintf( stderr, "bug [ocmp_simp_error].\n");
			exit( 0);
		}
		//t

		R_node_v2 = R_node_v1;
		while( R_node_v2 != NULL && R_node_v2->id2 != id_node_v->next->pos_id)
			R_node_v2 = R_node_v2->next;

		//t
		if( R_node_v2 == NULL)
		{
			fprintf( stderr, "bug [ocmp_simp_error].\n");
			exit( 0);
		}
		//t

		max_diff = calc_error( R_node_v1, R_node_v2);
		if( max_diff > error_v)
			error_v = max_diff;

		//
		R_node_v1 = R_node_v2->next;
		id_node_v = id_node_v->next;
	}

	return error_v;
}

/*
 *
 */
id_list_t* Span_Search( R_list_t* R_list_v, int W)
{
	float pivot, min_span;//, min_error;

	span_space_t* span_space_v;
	id_list_t* id_list_v, *id_list_v_min;

	MIN_HEAP_OPT = 0;

	//Initialize the search space.
	span_space_v = const_span_space( R_list_v);

	//The search process
	min_span = FLT_MAX;
	id_list_v_min = NULL;
	while( span_space_v->size >= 1)
	{
		//Find a pivot of the current search space.
		pivot = find_pivot( span_space_v);

		//Span affordability check.
		id_list_v = span_affordability_check( R_list_v, pivot);
		
		if( id_list_v->pos_id_n <= W)
		{
			//pivot is an affordable span.
			if( pivot < min_span)
			{
				min_span = pivot;

				id_list_release( id_list_v_min);
				
				id_list_v_min = id_list_v;

				id_list_v = NULL;
			}

			//Prune the search space appropriately.
			prune_span_space( span_space_v, pivot, -1);
		}
		else
		{
			//pivot is not an affordable span.
			//Prune the span space appropriately.
			prune_span_space( span_space_v, pivot, 1);
		}

		//
		id_list_release( id_list_v);
	}

	//Collect the error information of the simplified trajectory.
	//min_error = comp_simp_error( R_list_v, id_list_v_min);

	//Release the resources.
	span_space_release( span_space_v);

	//return min_error;
	return id_list_v_min;
}



//Approximate algorithms.
//
/*
 *	The improved implementation of Merge_sub
 *	with the heap structure.
 *
 *	It is possible that the R_array_v structure could be replaced by the R_list_t structure.
 *
 *	Re-design!
 *	Avoid the use the group_list_t structure.
 */
id_list_t* Merge_sub2( R_array_t* R_array_v, int W)
{
	int i, seg_n, b_cnt, top;
	float error_v;

	group_node_t* group_node_v1, *group_node_v2, *group_node_v;
	group_list_t* group_list_v;

	id_list_t* id_list_v;

	b_heap_t* b_h;

	if( R_array_v->size == 0)
		return id_list_alloc( );
	
	//Initialize the group_list_t struct.
	group_list_v = group_list_alloc( );

	seg_n = R_array_v->size;

	for( i=0; i<seg_n; i++)
	{
		append_group_list_entry( group_list_v, i, i);
	}

	//Initialize the b_heap.	
	b_h = alloc_b_heap( seg_n);

	b_cnt = 1;
	group_node_v = group_list_v->head->next;
	while( group_node_v->next != NULL)
	{
		b_h->obj_arr[ b_cnt].group_node_v = group_node_v;
		group_node_v->loc = b_cnt;

		//Calculate the merge error.
		error_v = calc_error( R_array_v->array[ group_node_v->sta_seg_id], 
			R_array_v->array[ group_node_v->next->end_seg_id]);

		b_h->obj_arr[ b_cnt].key = error_v;

		insert_b_heap( b_h, b_cnt ++);

		group_node_v = group_node_v->next;
	}
	
	//Merge iteratively.
	while( b_h->rear > 0)
	{
		//Finding the merge location.
		if( b_h->rear == 0)
			break;

		top = get_top( b_h);

		//if( b_h->obj_arr[ top].key > eps)
		if( group_list_v->size == W)
			break;
		
		//Perform the merge operation.
		group_node_v = b_h->obj_arr[ top].group_node_v;
		group_node_v1 = group_node_v->pre;
		group_node_v2 = group_node_v->next;

		//t
		if( group_node_v2 == NULL)
		{
			fprintf( stderr, "bug. [Merge_sub2]\n");
			exit( 0);
		}
		//t

		group_node_v1->next = group_node_v2;

		group_node_v2->sta_seg_id = group_node_v->sta_seg_id;

		group_node_v2->pre = group_node_v1;
		group_list_v->size --;

		free( group_node_v);

		//s
		emp_stat_v.memory_v -= sizeof( group_node_t);
		//s

		//Update the merge candidates accordingly.
		//The previous one if any.
		if( group_node_v1 != group_list_v->head)
		{
			error_v = calc_error( R_array_v->array[ group_node_v1->sta_seg_id],
								R_array_v->array[ group_node_v2->end_seg_id]);

			b_h->obj_arr[ group_node_v1->loc].key = error_v;

			update_key( b_h, group_node_v1->loc);
		}

		//The next one if any.
		if( group_node_v2->next != NULL)
		{
			error_v = calc_error( R_array_v->array[ group_node_v2->sta_seg_id],
								R_array_v->array[ group_node_v2->next->end_seg_id]);

			b_h->obj_arr[ group_node_v2->loc].key = error_v;

			update_key( b_h, group_node_v2->loc);
		}
	}
	
	//Construct the id_list_v from the group_list_v.
	id_list_v = id_list_alloc( );
	
	group_node_v = group_list_v->head->next;
	while( group_node_v != NULL)
	{
		append_id_list_entry( id_list_v, R_array_v->array[ group_node_v->sta_seg_id]->id1);

		group_node_v = group_node_v->next;
	}

	append_id_list_entry( id_list_v, R_array_v->array[ R_array_v->size-1]->id2);

	//Release the resource.
	group_list_release( group_list_v);

	release_b_heap( b_h);

	return id_list_v;
}


/*
 *	The implementation of the Merge algorithm.
 *	The packaging function.
 */
id_list_t* Merge( R_list_t* R_list_v, int W)
{
	R_array_t* R_array_v;
	id_list_t* id_list_v;

	MIN_HEAP_OPT = 1;

	R_array_v = R_array_transform( R_list_v);

	//id_list_v = Merge_sub2( R_array_v, eps);
	id_list_v = Merge_sub2( R_array_v, W);

	R_array_release( R_array_v);

	return id_list_v;
}

/*
 *
 */
void ini_part_node_dist( part_node_t* part_node_v)
{	
//	float time_cnt, time_interval;
	double dist, max_dist;
	
	R_node_t* iter;

	max_dist = 0;
	//part_node_v->pos_id_split = part_node_v->end->id1;
	part_node_v->split_loc = part_node_v->end;

	/*
	time_interval = 0;
	iter = part_node_v->sta;
	while( iter != NULL)
	{
		time_interval += iter->t_interval;

		if( iter == part_node_v->end)
			break;

		iter = iter->next;
	}
	*/

	//
	//time_cnt = 0;
	iter = part_node_v->sta;
	while( iter != part_node_v->end)
	{
		/*
		if( dist_opt == 1)	//Perpendicular distance.
			dist = calc_per_dist( iter->loc_v2, part_node_v->sta->loc_v1, part_node_v->end->loc_v2, PERPENDICULAR_OPTION);
		else				//Synchronous Euclidean distance.
		{
			//Compute the portion.
			time_cnt += iter->t_interval;

			dist = calc_syn_Euclidean_dist( iter->loc_v2, part_node_v->sta->loc_v1, part_node_v->end->loc_v2, time_cnt / time_interval);
		}
		*/

		dist = calc_angle( part_node_v->sta->loc_v1, part_node_v->end->loc_v2,
							iter->loc_v1, iter->loc_v2);
	
		if( dist > max_dist)
		{
			max_dist = dist;
			part_node_v->split_loc = iter;
		}

		iter = iter->next;
	}

	part_node_v->largest_dist = max_dist;
}

/*	
 *	The adaption of the Dougolas-Peucker algorithm (A split algorithm).
 */
id_list_t* Dougolas_Peucker_adapt( R_list_t* R_list_v, int W)
{
	int seg_n, b_cnt, top;
	double max_dist;

	part_list_t* part_list_v;
	part_node_t* part_node_v, *part_node_v1, *part_node_v2;

	b_heap_t* b_h;
	id_list_t* id_list_v;

	MIN_HEAP_OPT = 0;

	if( R_list_v->size == 0)
		return id_list_alloc( );

	seg_n = R_list_v->size;	

	//Initialize the part_list_t structure.
	part_list_v = part_list_alloc( );

	part_node_v = ( part_node_t*)malloc( sizeof( part_node_t));
	memset( part_node_v, 0, sizeof( part_node_t));

	part_node_v->sta = R_list_v->head->next;
	part_node_v->end = R_list_v->rear;

	ini_part_node_dist( part_node_v);
	part_list_v->head->next = part_node_v;
	part_list_v->rear = part_node_v;
	part_list_v->size ++;	

	//s
	emp_stat_v.memory_v += sizeof( part_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s
	
	//Initialize a binary heap.
	b_h = alloc_b_heap( seg_n + 2);

	b_cnt = 1;
	b_h->obj_arr[ b_cnt].part_node_v = part_node_v;
	b_h->obj_arr[ b_cnt].key = part_node_v->largest_dist;

	insert_b_heap( b_h, b_cnt ++);
	if( b_cnt > seg_n)
	{
		fprintf( stderr, "bug in [position_TS]\n");
		exit( 0);
	}

	//Partition iteratively.
	while( part_list_v->size < W - 1)
	{
		//Pick the part with the greatest distance measure.
		top = get_top( b_h);

		part_node_v1 = b_h->obj_arr[ top].part_node_v;

		if( b_h->obj_arr[ top].key == 0)
		{
			//No need to split the partition.
			break;
		}

		//Split the partition into two.
		part_node_v2 = ( part_node_t*)malloc( sizeof( part_node_t));
		memset( part_node_v2, 0, sizeof( part_node_t));
		
		part_node_v2->sta = part_node_v1->split_loc->next;
		part_node_v2->end = part_node_v1->end;
		part_node_v2->next = part_node_v1->next;

		part_node_v1->end = part_node_v1->split_loc;
		part_node_v1->next = part_node_v2;

		if( part_list_v->rear == part_node_v1)
			part_list_v->rear = part_node_v2;			

		part_list_v->size ++;

		//s
		emp_stat_v.memory_v += sizeof( part_node_t);
		if( emp_stat_v.memory_v > emp_stat_v.memory_max)
			emp_stat_v.memory_max = emp_stat_v.memory_v;
		//s

		//Re-set the attributes.
		ini_part_node_dist( part_node_v1);
		ini_part_node_dist( part_node_v2);

		//Insertion to the heap.
		//update_key( b_h, top);
		b_h->obj_arr[ top].key = part_node_v1->largest_dist;
		insert_b_heap( b_h, top);
		
		b_h->obj_arr[ b_cnt].part_node_v = part_node_v2;
		b_h->obj_arr[ b_cnt].key = part_node_v2->largest_dist;
		insert_b_heap( b_h, b_cnt ++);

		if( b_cnt > seg_n + 1)
		{
			fprintf( stderr, "bug in [position_TS]\n");
			exit( 0);
		}
	}

	//Collect the id_list_t information.
	id_list_v = id_list_alloc( );
	
	max_dist = 0;
	part_node_v = part_list_v->head->next;
	while( part_node_v != 0)
	{
		append_id_list_entry( id_list_v, part_node_v->sta->id1);

		if( part_node_v->largest_dist > max_dist)
			max_dist = part_node_v->largest_dist;

		part_node_v = part_node_v->next;
	}

	//Append the last position.
	append_id_list_entry( id_list_v, part_list_v->rear->end->id2);


	//Release the resources.
	release_b_heap( b_h);
	part_list_release( part_list_v);

	//t
	//printf( "Distance-based error: %lf\n", max_dist);
	//t

	return id_list_v;
}

/*
 *	The interface of the algorithms for the Min-Error problem.
 */
float DPTS_v2( tra_list_t* tra_list_v, int W, int alg_opt)
{
	//int simp_size, quota;
	//float tmp1, tmp2;
	float error_v;
    FILE* s_fp;
	
	R_list_t* R_list_v;//, *R_list_v_appr, *R_list_v_appr_pos_per, *R_list_v_appr_pos_syn, *R_list_v_appr_pos_per_2, *R_list_v_appr_pos_syn_2;
	id_list_t* id_list_v;//, *id_list_v_pos_per, *id_list_v_pos_syn, *id_list_v_pos_per_2, *id_list_v_pos_syn_2;

	id_list_v = NULL;
	R_list_v = NULL;

	R_list_v = R_list_transform( tra_list_v);
	if( R_list_v == NULL)
		//return tra_list_v->pos_n;
		return 0;

	/*t/
	print_error_matrix( R_list_v, 2);
	/*t*/

	if( alg_opt == 1)
	{
		printf( "the basic DP-Error ...\n");
		//simp_size = DP_Number( R_list_v, eps, 0);
		error_v = DP_Error( R_list_v, W, 1);

		//simp_size += 1;
	}
	else if( alg_opt == 2)
	{
		printf( "the enhanced DP-Error ...\n");
		//simp_size = DP_Number( R_list_v, eps, 1);
		error_v = DP_Error( R_list_v, W, 2);

		//simp_size += 1;	//
	}
	else if( alg_opt == 3)
	{
		printf( "Error-Search ...\n");
		//simp_size = SP_Method_B( R_list_v, eps);
		id_list_v = Error_Search( R_list_v, W);

		//simp_size += 1;	//SP_Method_7 returns the length of the SP.
	}
	else if( alg_opt == 4)
	{
		printf( "Span-Search ...\n");
		//id_list_v = SP_Method_prac( R_list_v, eps);	
		//id_list_v = Error_Search( R_list_v, W);
		id_list_v = Span_Search( R_list_v, W);
	}
	else if( alg_opt == 5)
	{
		printf( "Split ...\n");
		id_list_v = Dougolas_Peucker_adapt( R_list_v, W);

		//simp_size += 1;	//SP_Method_B returns the length of the SP.
	}
	//else if( alg_opt == 6)
	//{
	//	printf( "Merge ...\n");
		//id_list_v = SP_Method_prac_and_theo( R_list_v, eps);
	//	id_list_v = Merge( R_list_v, W);
	//}
	else// if( alg_opt == 7)
	{
		printf( "Error-Search [appro] ...\n");
		//simp_size = DP_Number( R_list_v, eps, 1);
		error_v = Error_Search_DouglasPeucker( R_list_v, W);
	}
		
	
	/*
	if( id_list_v != NULL)
	{
		//t
		id_list_print( id_list_v);
		printf( "\n");
		//t

		printf( "Collecting statistics ...\n");

		simp_size = id_list_v->pos_id_n - 1;

		//Collect other statistics (length, speed and position).		
		//Construct another R_list_v.
		R_list_v_appr = R_list_transform_appr( tra_list_v, id_list_v);

		
		collect_length_and_speed_error( R_list_v, R_list_v_appr, &emp_stat_v.length_err, &emp_stat_v.speed_err);
		collect_position_error( R_list_v, R_list_v_appr, &emp_stat_v.pos_err_per, 1);
		collect_position_error( R_list_v, R_list_v_appr, &emp_stat_v.pos_err_syn, 2);

		
		collect_direction_error( R_list_v, R_list_v_appr, &emp_stat_v.dir_err_dpts);

		emp_stat_v.pos_err_per_bound = collect_position_error_bound( R_list_v_appr, eps, 1);
		emp_stat_v.pos_err_syn_bound = collect_position_error_bound( R_list_v_appr, eps, 2);

		//Collect the position errors of existing position-preserving trajectory simplification algorithms.
		//Optimal method: the DP method.
		quota = id_list_v->pos_id_n - 1;
		//quota = R_list_v->size + 1;

		emp_stat_v.pos_err_per_opti = position_TS_DP( R_list_v, quota, 1);
		emp_stat_v.pos_err_syn_opti = position_TS_DP( R_list_v, quota, 2);

		id_list_v_pos_per = SP_Method_prac_dist( R_list_v, emp_stat_v.pos_err_per_opti, 1);
		id_list_v_pos_syn = SP_Method_prac_dist( R_list_v, emp_stat_v.pos_err_syn_opti, 2);

		R_list_v_appr_pos_per = R_list_transform_appr( tra_list_v, id_list_v_pos_per);
		R_list_v_appr_pos_syn = R_list_transform_appr( tra_list_v, id_list_v_pos_syn);
		
		//t
		collect_position_error( R_list_v, R_list_v_appr_pos_per, &tmp1, 1);
		collect_position_error( R_list_v, R_list_v_appr_pos_syn, &tmp2, 2);

		if( emp_stat_v.pos_err_per_opti != tmp1 ||
			emp_stat_v.pos_err_syn_opti != tmp2)
		{
			fprintf( stderr, "pos_err_per_opti or pos_err_syn_opti inconsistency. [DPTS]\n");
			exit( 0);
		}

		collect_direction_error( R_list_v, R_list_v_appr_pos_per, &emp_stat_v.dir_err_per_opti);
		collect_direction_error( R_list_v, R_list_v_appr_pos_syn, &emp_stat_v.dir_err_syn_opti);

		//Heuristic method: Dogoulas-Peucker.
		id_list_v_pos_per_2 = position_TS( R_list_v, quota, 1);
		id_list_v_pos_syn_2 = position_TS( R_list_v, quota, 2);

		R_list_v_appr_pos_per_2 = R_list_transform_appr( tra_list_v, id_list_v_pos_per_2);
		R_list_v_appr_pos_syn_2 = R_list_transform_appr( tra_list_v, id_list_v_pos_syn_2);

		collect_position_error( R_list_v, R_list_v_appr_pos_per_2, &emp_stat_v.pos_err_per_heur, 1);
		collect_position_error( R_list_v, R_list_v_appr_pos_syn_2, &emp_stat_v.pos_err_syn_heur, 2);

		//s
		printf( "pos_err_per:\t\t%f\npos_err_per_heur:\t%f\npos_err_per_opti:\t%f\npos_err_per_bound:\t%f\n\n", 
			emp_stat_v.pos_err_per, emp_stat_v.pos_err_per_heur, emp_stat_v.pos_err_per_opti, emp_stat_v.pos_err_per_bound);
		printf( "pos_err_syn:\t\t%f\npos_err_syn_heur:\t%f\npos_err_syn_opti:\t%f\npos_err_syn_bound:\t%f\n\n", 
			emp_stat_v.pos_err_syn, emp_stat_v.pos_err_syn_heur, emp_stat_v.pos_err_syn_opti, emp_stat_v.pos_err_syn_bound);

		collect_direction_error( R_list_v, R_list_v_appr_pos_per_2, &emp_stat_v.dir_err_per_heur);
		collect_direction_error( R_list_v, R_list_v_appr_pos_syn_2, &emp_stat_v.dir_err_syn_heur);


		//s
		printf( "%f\n%f\n%f\n%f\n%f\n\n", emp_stat_v.dir_err_dpts, 
									emp_stat_v.dir_err_per_heur, emp_stat_v.dir_err_per_opti,
									emp_stat_v.dir_err_syn_heur, emp_stat_v.dir_err_syn_opti);
	
	}
	*/
	
	/*
	printf( "Original size: %i\n", R_list_v->size + 1);
	printf( "Simplified size: %i\n", simp_size);
	printf( "Compression ratio: %f\n\n", ( float)simp_size / ( R_list_v->size + 1));
	*/

	/*
	if( id_list_v != NULL)
	{
		R_list_release( R_list_v_appr);

		id_list_release( id_list_v_pos_per);
		id_list_release( id_list_v_pos_syn);

		id_list_release( id_list_v_pos_per_2);
		id_list_release( id_list_v_pos_syn_2);

		R_list_release( R_list_v_appr_pos_per);
		R_list_release( R_list_v_appr_pos_syn);

		R_list_release( R_list_v_appr_pos_per_2);
		R_list_release( R_list_v_appr_pos_syn_2);
	}
	*/

	if( id_list_v != NULL) {
        error_v = comp_simp_error(R_list_v, id_list_v);
        if( ( s_fp = fopen( RES_FILE, "w")) == NULL)
        {
            fprintf( stderr, "Cannot open the res.txt\n");
            exit( 0);
        }
        int	pos_id_n = id_list_v->pos_id_n;
        id_node_t* p = id_list_v->head->next;
        for(int i=0;i<pos_id_n;i++){
            fprintf( s_fp, "%d\n", p->pos_id);
            p = p->next;
        }

    }

	id_list_release( id_list_v);
	R_list_release( R_list_v);

	return error_v;
}

/*
 *	Empirical study for the Min-Error problem.
 */
void emp_DPTS_v2( )
{
	int ori_size;
	float error_v;

	emp_config_t* emp_config_v;
//	id_list_t* id_list_v;
	tra_list_t* tra_list_v;

	FILE* s_fp;

	//Pre-set the statistics collector emp_stat_v.
	memset( &emp_stat_v, 0, sizeof( emp_stat_t));
	emp_stat_v.C.min = FLT_MAX;

	//Read the config.
	emp_config_v = emp_read_config( );

	//precision_thr_v2 = emp_config_v->precision_thr;
	precision_thr_v2 = PRECISION_THR;

	//Read the data.
	//printf( "Reading data ...\n");
	if( emp_config_v->dataset_tag == 1)
	{
		emp_config_v->line_ignore_n = 6;
		tra_list_v = emp_read_data_v1( emp_config_v);
	}
	else
	{
		emp_config_v->line_ignore_n = 0;
		tra_list_v = emp_read_data_v2( emp_config_v);
	}

	//
	ori_size = tra_list_v->pos_n;
	if( TRA_SIZE_TRUNCATION_OPT)
		tra_list_v->pos_n = comp_bounded_power2( tra_list_v->pos_n - 1);
	//printf( "Original size: %i\n", tra_list_v->pos_n);

	emp_config_v->W = int ( emp_config_v->W_percent * tra_list_v->pos_n);

#ifndef WIN32
	rusage before_simp_t;
	get_cur_time( &before_simp_t);
#endif
	
	//Perform the simplification process.
	printf( "Simplifying ...\n");
	//simp_size = DPTS( tra_list_v, emp_config_v->eps, emp_config_v->alg_opt);

	if( emp_config_v->alg_opt == 0)
	{
		printf( "the method of HWT ...\n");
		error_v = emp_TS_HWT_v2( tra_list_v, emp_config_v->W);
	}
	else
		error_v = DPTS_v2( tra_list_v, emp_config_v->W, emp_config_v->alg_opt);
	
	/*t/
	R_list_t* R_list_v = R_list_transform( tra_list_v);
	position_TS( R_list_v, R_list_v->size / 2);
	/*t*/
	printf( "Finished!\n");

#ifndef WIN32
	rusage after_simp_t;
	
	get_cur_time( &after_simp_t);
	collect_time( &before_simp_t, &after_simp_t, &emp_stat_v.r_time);
#endif

	//Collect the statistics.

	//cmp_ratio.
	//emp_stat_v.cmp_ratio = ( float)simp_size / emp_config_v->pos_n;

	//percent.
	//emp_stat_v.percent = emp_stat_v.eps_n / ( emp_config_v->pos_n * ( emp_config_v->pos_n - 1) / 2);

	//C.aver.
	//if( emp_stat_v.cap_n != 0)
	//	emp_stat_v.C.aver = emp_stat_v.C.sum / emp_stat_v.cap_n;

	//Print the results & statistics.
	if( ( s_fp = fopen( STATISTIC_FILE, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open the stat.txt\n");
		exit( 0);
	}

	//Simplification results.
	//fprintf( s_fp, "%i\n%i\n%f\n\n", tra_list_v->pos_n, simp_size, emp_stat_v.cmp_ratio);

	//Statistics.
	//fprintf( s_fp, "%i\n\n", tra_list_v->pos_n);
	fprintf( s_fp, "%f\n\n", error_v);
	fprintf( s_fp, "%f\n%f\n\n", emp_stat_v.r_time, emp_stat_v.memory_max / (1024 * 1024));

	//fprintf( s_fp, "%f\n", emp_stat_v.error_aver);

	//fprintf( s_fp, "%f\n\n", emp_stat_v.percent);

	//if( emp_stat_v.C.min == FLT_MAX)
	//	emp_stat_v.C.min = 1;
	//fprintf( s_fp, "%f\n%f\n%f\n\n", emp_stat_v.C.min, emp_stat_v.C.max, emp_stat_v.C.aver);

	
	//emp_stat_v.length_err_bound = cos( emp_config_v->eps);
	//emp_stat_v.speed_err_bound = cos( emp_config_v->eps);

	//fprintf( s_fp, "%f\n%f\n\n%f\n%f\n\n", emp_stat_v.length_err, emp_stat_v.length_err_bound, 
	//	emp_stat_v.speed_err, emp_stat_v.speed_err_bound);
	
	//fprintf( s_fp, "%f\n%f\n%f\n%f\n\n%f\n%f\n%f\n%f\n\n", 
	//	emp_stat_v.pos_err_per, emp_stat_v.pos_err_per_heur, emp_stat_v.pos_err_per_opti, emp_stat_v.pos_err_per_bound,
	//	emp_stat_v.pos_err_syn, emp_stat_v.pos_err_syn_heur, emp_stat_v.pos_err_syn_opti, emp_stat_v.pos_err_syn_bound);

	//fprintf( s_fp, "%f\n%f\n%f\n%f\n%f\n\n", emp_stat_v.dir_err_dpts, 
	//								emp_stat_v.dir_err_per_heur, emp_stat_v.dir_err_per_opti, 
	//								emp_stat_v.dir_err_syn_heur, emp_stat_v.dir_err_syn_opti);

	//appro_f (should be executed lastly in order not to affect other statistics).
	/*
	if( emp_config_v->alg_opt > 5 && emp_config_v->appr_b_tag == 1)
	{
		printf( "Optimal simplification ...\n");
		simp_size_t = DPTS( tra_list_v, emp_config_v->eps, 5);
		printf( "Finished again!\n");
		
		emp_stat_v.appro_f = ( float)simp_size / simp_size_t;
	}
	else
		emp_stat_v.appro_f = 1;
	*/

	//fprintf( s_fp, "%f\n\n", emp_stat_v.appro_f);


	//release the resources.
	fclose( s_fp);
	tra_list_v->pos_n = ori_size;
	tra_list_release( tra_list_v);
	free( emp_config_v);

	//printf( "Minimized error: %f\n", error_v);
	//printf( "Average error: %f\n", emp_stat_v.error_aver);
	//printf( "Running time: %f\n", emp_stat_v.r_time);
	//printf( "Number of checkings: %f\n", emp_stat_v.eps_n);
	//printf( "Memory balance: %f\n", emp_stat_v.memory_v);
}


#ifndef WIN32
/*
 *	Compress each trajectory in the dataset.
 *
 *	For the Min-Error problem.
 */
void compress_data_set_v2( )
{
	int d_1, d_2, d_3, i, folder_num, folder_cnt, file_cnt, file_n_thr;
	//float ori_pos_cnt, simp_pos_cnt;
	float error_v, max_error_v;
	//float eps[ 19] = {0, 0.001, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3};
	float W[ 18] = {0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};

	char folder_name[ MAX_FILENAME_LENG];
	char dir_path[ MAX_PATHNAME_LENG];
	char f_name[ MAX_FILENAME_LENG];

	DIR* dir_p;
	struct dirent* dir_r;
	unsigned char is_file =0x8;

	tra_list_t* tra_list_v;


	FILE* s_fp;

	if( ( s_fp = fopen( COMPRESSION_STAT, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open %s.\n", COMPRESSION_STAT);
		exit( 0);
	}

	/*t*/
	printf( "# of folders to simplify:\n");
	scanf( "%i", &folder_num);
	printf( "the threshold of the # of files:\n");
	scanf( "%i", &file_n_thr);
	/*t*/
	
	
	for( i = 0; i < 18; i++)
	{
		//Traverse the eps setting.
		//ori_pos_cnt = 0;
		//simp_pos_cnt = 0;
		
		folder_cnt = 0;

		//
		max_error_v = 0;
		for( d_1 = 0; d_1 <= 9; d_1 ++)
		{
			for( d_2 = 0; d_2 <= 9; d_2 ++)
			{
				for( d_3 = 3; d_3 <= 14; d_3 ++)
				{
					folder_cnt ++;
					if( folder_cnt > folder_num)
						goto E;

					sprintf( folder_name, "%i%i%i", d_1, d_2, d_3);
					sprintf( dir_path, "%s%s%s", "./data/Geolife/Data/", folder_name, "/Trajectory/");

					//sprintf( folder_name, "%i%i", d_2, d_3);
					//sprintf( dir_path, "%s%s%s", "./data/T-Drive/", folder_name, "/");
					
					//t
					//printf( "%s\t%f\n", dir_path, eps[ i]);
					//t
					
					//Handle the current folder.
					if( ( dir_p = opendir( dir_path)) == NULL)
					{
						fprintf( stderr, "Cannot open %s.\n", dir_path);
						exit( 0);
					}
					
					file_cnt = 0;
					while( ( dir_r = readdir( dir_p)) != NULL)
					{
						//
						if( dir_r->d_type == is_file)
						{
							file_cnt ++;
							if( file_cnt > file_n_thr)
								break;

							sprintf( f_name, "%s%s", dir_path, dir_r->d_name);

							/*t*/
							printf( "%s\n", f_name);
							/*t*/
						
							//Read the trajectory.
							tra_list_v = read_trajectory_v1( f_name);

							//ori_pos_cnt += tra_list_v->pos_n;

							//Perform the simplification.
							//simp_pos_cnt += DPTS( tra_list_v, eps[ i], 5);
							error_v = DPTS_v2( tra_list_v, ( int)( W[ i] * tra_list_v->pos_n), 3);

							if( error_v > max_error_v)
								max_error_v = error_v;

							//Release the resources.
							tra_list_release( tra_list_v);																			
						}
						
					}//while( dir_r)		
					
					closedir( dir_p);
				}//for(d_3)
			}//for( d_2)
		}//for( d_1)
E:
		//fprintf( s_fp, "%f\t%f\t%f\t%f\n", eps[ i], ori_pos_cnt, simp_pos_cnt, simp_pos_cnt / ori_pos_cnt);
		fprintf( s_fp, "%f\t%f\n", W[ i], max_error_v);
	}//for( i)

	fclose( s_fp);		
}

#endif

//Added on April 17, 2014.
/*
 *	The Trajectory simplification interface based on HWT.
 *	
 *	The x & y coordinates are compressed via HWT.
 *	
 *	@return the simplification error;
 */
float emp_TS_HWT( tra_list_t* tra_list_v, int W)
{
	float error_v;
	tra_list_t* tra_list_v2;
	wavelet_data_t* wavelet_data_v1, *wavelet_data_v2, *wavelet_data_v3, *wavelet_data_v4;

	//Construct two wavelet_data_t structures;
	tra_list_to_wavelet_data( tra_list_v, wavelet_data_v1, wavelet_data_v2);

	//Simplify the trajectory via HWT;
	wavelet_data_v3 = TS_HWT( wavelet_data_v1, W);		//
	wavelet_data_v4 = TS_HWT( wavelet_data_v2, W);

	/*t/
	print_wavelet_data( wavelet_data_v1, "test.txt");
	print_wavelet_data( wavelet_data_v3, "test.txt");
	print_wavelet_data( wavelet_data_v2, "test.txt");
	print_wavelet_data( wavelet_data_v4, "test.txt");
	/*t*/

	wavelet_data_to_tra_list( wavelet_data_v3, wavelet_data_v4, tra_list_v2);
	
	//Compute the error between two tra_list_v's;
	error_v = calc_error_tra( tra_list_v, tra_list_v2);

	//Release the resource;
	tra_list_release( tra_list_v2);
	wavelet_data_release( wavelet_data_v1);
	wavelet_data_release( wavelet_data_v2);
	wavelet_data_release( wavelet_data_v3);
	wavelet_data_release( wavelet_data_v4);

	return error_v;
}

/*
 *	Perform HWT on a wavelet_data_t structure;
 */
wavelet_data_t* TS_HWT( wavelet_data_t* wavelet_data_v, int k)
{
	wavelet_data_t* wavelet_data_v2;
	wavelet_coeff_t* wavelet_coeff_v;

	//
	wavelet_coeff_v = HWT( wavelet_data_v, k, 1);
	
	//
	wavelet_data_v2 = HWDT( wavelet_coeff_v);

	//Release the resource.
	wavelet_coeff_release( wavelet_coeff_v);

	return wavelet_data_v2;
}

/*
 *	Calculate the directional error between two trajectories in the form of tra_list_t;
 */
float calc_error_tra( tra_list_t* tra_list_v1, tra_list_t* tra_list_v2)
{
	int cnt, seg_n, slope_cnt;
	float error_max, angle_diff;
	R_list_t* R_list_v1, *R_list_v2;
	R_node_t* R_node_v1, *R_node_v2;

	seg_n = tra_list_v1->pos_n - 1;

	//
	//precision_thr_v2 = 0.0002;

	//
	R_list_v1 = R_list_transform( tra_list_v1);
	R_list_v2 =	R_list_transform( tra_list_v2);

	slope_cnt = 0;
	cnt = 0;
	error_max = 0;
	R_node_v1 = R_list_v1->head->next;
	R_node_v2 = R_list_v2->head->next;
	while( cnt < seg_n)
	{
		//pre-check 
		if( R_node_v1->slope_tag != 1 && R_node_v2->slope_tag != 1)
		{
			//Both slopes could be used;
			angle_diff = calc_angular_diff( R_node_v1->slope, R_node_v2->slope);

			//t
			//printf( "%f  %f  %f\n", R_node_v1->slope, R_node_v2->slope, angle_diff);
			//t
		
			if( angle_diff > error_max)
				error_max = angle_diff;

			//s
			slope_cnt ++;
			emp_stat_v.error_aver += angle_diff;
			//s
		}

		R_node_v1 = R_node_v1->next;
		R_node_v2 = R_node_v2->next;

		cnt++;
	}

	//Release the resource;
	R_list_release( R_list_v1);
	R_list_release( R_list_v2);

	//s
	if( slope_cnt != 0)
		emp_stat_v.error_aver /= slope_cnt;
	
	return error_max;
}

/*
 *	Counterpart of calc_error_tra;
 */	
float calc_error_R( R_list_t* R_list_v1, R_list_t* R_list_v2)
{
	float error_max, angle_diff;
	R_node_t* R_node_v1, *R_node_v2;

	R_node_v1 = R_list_v1->head->next;
	R_node_v2 = R_list_v2->head->next;

	error_max = 0;
	while( R_node_v1 != NULL && R_node_v2 != NULL)
	{
		angle_diff = calc_angular_diff( R_node_v1->slope, R_node_v2->slope);

		if( angle_diff > error_max)
			error_max = angle_diff;

		R_node_v1 = R_node_v1->next;
		R_node_v2 = R_node_v2->next;

		//s
		emp_stat_v.error_aver += angle_diff;
		//s
	}

	//s
	if( R_list_v1->size != 0)
		emp_stat_v.error_aver /= R_list_v1->size;
	//s

	return error_max;
}

/*
 *	The implementation of the Error_Search algorithm adopting the Douglas Peucker algorithm
 *	for the feasibility check purpose.
 */
float Error_Search_DouglasPeucker( R_list_t* R_list_v, int W)
{
	int pos_n, size, left, right, mid, rtn;
	//float min_affordable_error;
	float error_v;
	float* error_array_v;

	error_space_t* error_space_v;
	//id_list_t* id_list_v, *id_list_v_min;

	//
	pos_n = R_list_v->size + 1;
	size = pos_n * ( pos_n - 1) / 2;

	//
	error_space_v = const_error_space_v2( R_list_v);

	error_array_v = reform_error_space( error_space_v);

	qsort( error_array_v, size, sizeof( float), compare);

	//Binary search;
	//min_affordable_error = FLT_MAX;
	//id_list_v_min = NULL;
	left = 0;
	right = size - 1;
	while( left < right)
	{
		mid = ( left + right) / 2;

		//id_list_v = SP_Method_prac_and_theo( R_list_v, error_array_v[ mid]);
		rtn = Split( R_list_v, error_array_v[ mid]);

		if( rtn <= W)
		{
			right = mid;
	
			//id_list_release( id_list_v_min);

			//id_list_v_min = id_list_v;

			//id_list_v = NULL;
		}
		else
			left = mid + 1;
		
		//id_list_release( id_list_v);
	}
	
	//min_affordable_error = error_array_v[ right];
	error_v = error_array_v[ right];

	//Release the resources.
	error_space_release( error_space_v);
	free( error_array_v);

	//s
	emp_stat_v.memory_v -= size * sizeof( float);
	//s

	//return min_affordable_error;
	//return id_list_v_min;
	return error_v;
}

/*
 *	The Trajectory simplification interface based on HWT.
 *	
 *	the (direction, length) pairs are compressed via HWT;
 *	
 *	@return the simplification error;
 */
float emp_TS_HWT_v2( tra_list_t* tra_list_v, int W)
{
	float error_v;
	wavelet_data_t* wavelet_data_v1, *wavelet_data_v2, *wavelet_data_v3, *wavelet_data_v4;
	R_list_t* R_list_v1, *R_list_v2;

	//
	tra_list_v->pos_n = tra_list_v->pos_n + 1;
	R_list_v1 = R_list_transform( tra_list_v);

	//Construct two wavelet_data_t structures;
	R_list_to_wavelet_data( R_list_v1, wavelet_data_v1, wavelet_data_v2);

	//Simplify the trajectory via HWT;
	wavelet_data_v3 = TS_HWT( wavelet_data_v1, W);//
	wavelet_data_v4 = TS_HWT( wavelet_data_v2, W);

	/*t/
	printf( "%i\n", W);
	print_wavelet_data( wavelet_data_v1, "test.txt");
	print_wavelet_data( wavelet_data_v3, "test.txt");
	//print_wavelet_data( wavelet_data_v2, "test.txt");
	//print_wavelet_data( wavelet_data_v4, "test.txt");
	/*t*/

	wavelet_data_to_R_list( wavelet_data_v3, wavelet_data_v4, R_list_v2);
	
	//Compute the error between two tra_list_v's;
	//error_v = calc_error_tra( tra_list_v, tra_list_v2);
	error_v = calc_error_R( R_list_v1, R_list_v2);

	//Release the resource;
	//tra_list_release( tra_list_v2);
	R_list_release( R_list_v1);
	R_list_release( R_list_v2);
	wavelet_data_release( wavelet_data_v1);
	wavelet_data_release( wavelet_data_v2);
	wavelet_data_release( wavelet_data_v3);
	wavelet_data_release( wavelet_data_v4);

	return error_v;
}
