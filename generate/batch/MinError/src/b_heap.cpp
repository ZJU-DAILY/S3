/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */


#include "b_heap.h"



//-------------------------------------

/*
int cmp( const void* obj_v1, const void* obj_v2)
{
	return (*(h_obj_t**)obj_v1)->v <  (*(h_obj_t**)obj_v2)->v ? 1 : 
	(*(h_obj_t**)obj_v1)->v > (*(h_obj_t**)obj_v2)->v ? -1 : 0
}
*/

/*
*	cmp_min indicates the comparison functon for the min-heap.
*/
int cmp_min( int* array, int n1, int n2, h_obj_t* obj_v)
{
	return  obj_v[array[n1]].key < obj_v[array[n2]].key ? 1 :
	obj_v[array[n1]].key > obj_v[array[n2]].key ? -1 : 0;
}

/*
*	cmp_max indicates the comparison function for the maximum-heap..
*/
int cmp_max( int* array, int n1, int n2, h_obj_t* obj_v)
{
	return  obj_v[array[n1]].key < obj_v[array[n2]].key ? -1 :
	obj_v[array[n1]].key > obj_v[array[n2]].key ? 1 : 0;
}

/*
 *	Allocate a binary heap.
 */
b_heap_t* alloc_b_heap( int size)
{
	b_heap_t* b_h;

	b_h = ( b_heap_t*)malloc( sizeof( b_heap_t));
	memset( b_h, 0, sizeof( b_heap_t));

	b_h->size = size + 1;		//Modified by Cheng, 20 July, 2013.

	b_h->h_arr = ( int*)malloc( b_h->size * sizeof( int));
	memset( b_h->h_arr, 0, b_h->size * sizeof( int));

	b_h->obj_arr = ( h_obj_t*)malloc( b_h->size * sizeof( h_obj_t));
	memset( b_h->obj_arr, 0, b_h->size * sizeof( h_obj_t));

	//s
	b_h->m_size += sizeof( b_heap_t) + ( sizeof( int) + sizeof( h_obj_t)) * b_h->size;

	emp_stat_v.memory_v += sizeof( b_heap_t) + ( sizeof( int) + sizeof( h_obj_t)) * b_h->size;
	
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	return b_h;
}

/*
 *	Release a binary heap.
 */
void release_b_heap( b_heap_t* b_h)
{
	//s
	emp_stat_v.memory_v -= sizeof( b_heap_t) + 
		( sizeof( int) + sizeof( h_obj_t)) * b_h->size;
	//s

	free( b_h->obj_arr);
	free( b_h->h_arr);
	free( b_h);
	b_h = NULL;
}


/*
*	b_t_heapify adjusts the heap from the "bottum" to the "top".
*/
void b_t_heapify( int* array, int cur, h_obj_t* obj_v)
{
	int	n_tmp;

	//Top point or the property is satisfied.
	if( MIN_HEAP_OPT)	//min-heap.
	{
		if( cur <= 1 || cmp_min( array, cur, cur/2, obj_v) <= 0)
		return;
	}
	else				//max-heap.
	{
		if( cur <= 1 || cmp_max( array, cur, cur/2, obj_v) <= 0)
			return;
	}

	//Do the arrangement.
	n_tmp = array[cur];
	array[cur] = array[cur/2];
	obj_v[array[cur/2]].loc = cur;
	array[cur/2] = n_tmp;
	obj_v[n_tmp].loc = cur/2;

	//Iterate.
	b_t_heapify( array, cur/2, obj_v);
}


/*
*	t_b_heapify adjusts the heap from the "bottum" to the "top".
*/
void t_b_heapify( int* array, int cur, int rear, h_obj_t* obj_v)
{
	int n_tmp, n_swap, l_c = 2*cur, r_c = 2*cur + 1;
	if( l_c > rear)
		return;

	//Do the arrangement.
	n_swap = l_c;
	if( MIN_HEAP_OPT)	//min-heap.
	{
		if( r_c <= rear && cmp_min( array, r_c, l_c, obj_v) > 0)
			n_swap = r_c;
		
		if( cmp_min( array, cur, n_swap, obj_v) >= 0)
			return;
	}
	else				//max-heap.
	{
		if( r_c <= rear && cmp_max( array, r_c, l_c, obj_v) > 0)
			n_swap = r_c;
		
		if( cmp_max( array, cur, n_swap, obj_v) >= 0)
			return;
	}

	//Do the arrangement.
	n_tmp = array[n_swap];
	array[n_swap] = array[cur];
	obj_v[array[cur]].loc = n_swap;
	array[cur] = n_tmp;
	obj_v[n_tmp].loc = cur;

	//Iterate.
	t_b_heapify( array, n_swap, rear, obj_v);
}


/*
*	Insert a node into the heap.
*
*	@b_h indicates the heap.
*	@n indicates the object to be inserted by "obj_v[n]".
*
*	@return 0 if un-successful; otherwise, return 1.
*/
int insert_b_heap( b_heap_t* b_h, int n)
{
	b_h->rear ++;
	if( b_h->rear > b_h->size)
	{
		fprintf( stderr, "Heap overflows.\n");
		return 0;
	}

	//Insert the element at the rear position.
	b_h->h_arr[b_h->rear] = n;
	b_h->obj_arr[n].loc = b_h->rear;
	
	//Do the arrangement[buttom-to-top].
	b_t_heapify( b_h->h_arr, b_h->rear, b_h->obj_arr);

	return 1;
}


/*	
*	get_top achieves the top node of the heap.
*/
int get_top( b_heap_t* b_h)
{
	int n_tmp, rtn;
	
	if( b_h->rear == 0)
	{
		fprintf( stderr, "Empty queue.\n");
		exit(0);
	}
	
	rtn= b_h->h_arr[1];
	
	if(b_h->rear == 1)
	{
		b_h->rear = 0;
		return rtn;
	}

	//Replace the top the rear element.
	b_h->h_arr[1] = b_h->h_arr[b_h->rear];
	n_tmp = b_h->h_arr[1];
	b_h->obj_arr[n_tmp].loc = 1; 
	b_h->rear --;

	//Do the arrangement[top-to-buttom].
	t_b_heapify( b_h->h_arr, 1, b_h->rear, b_h->obj_arr);

	return rtn;
}

/*
*	update_key updates the value of a specific object indexed by the heap.
*/
void update_key( b_heap_t* b_h, int k)
{
	int cur, p, l_c, r_c;
	cur = b_h->obj_arr[k].loc;
	p = cur / 2;
	l_c = 2 * cur;
	r_c = 2 * cur + 1;
	
	if( MIN_HEAP_OPT)	//min-heap.
	{
		if( p >= 1 && cmp_min( b_h->h_arr, cur, p, b_h->obj_arr) > 0)
			b_t_heapify( b_h->h_arr, cur, b_h->obj_arr);
		else	//p < 1 || cmp_max( ) <= 0;
			t_b_heapify( b_h->h_arr, cur, b_h->rear, b_h->obj_arr);
	}
	else				//max-heap.
	{
		if( p >= 1 && cmp_max( b_h->h_arr, cur, p, b_h->obj_arr) > 0)
			b_t_heapify( b_h->h_arr, cur, b_h->obj_arr);
		else	//p < 1 || cmp_max( ) <= 0;
			t_b_heapify( b_h->h_arr, cur, b_h->rear, b_h->obj_arr);
	}

	return;
}

/*
*	print_b_heap prints the whole heap.
*/
void print_b_heap( FILE* o_fp, b_heap_t* b_h)
{
	int i, cnt = 0, n_col = 1;
	for( i=1; i<=b_h->rear; i++)
	{
		fprintf( o_fp, "%f  ", b_h->obj_arr[b_h->h_arr[i]].key);
		cnt ++;
		if( cnt == 2*n_col - 1)
		{
			fprintf( o_fp, "\n");
			n_col *= 2;
		}
	}
}

/*
*	test_b_heap verifys the implementation of the bianry_heap.
*/
void test_b_heap( )
{
	int i, l_max, top;
	char i_file[MAX_FILENAME_LENG], o_file[MAX_FILENAME_LENG];
	FILE* c_fp, *i_fp, *o_fp;

	if( (c_fp = fopen( B_CONFIG_FILE, "r")) == NULL)
	{
		fprintf( stderr, "The b_config.txt file cannot be opened.\n");
		exit( 0);
	}

	//read the configuration info.
	fscanf( c_fp, "%i%s%s", &l_max, i_file, o_file);
	if(	(i_fp = fopen( i_file, "r")) == NULL ||
		(o_fp = fopen( o_file, "w")) == NULL)
	{
		fprintf( stderr, "The input/output file cannot be opened.\n");
		exit( 0);
	}

	//allocate the main memory.
	b_heap_t* b_h = (b_heap_t*)malloc( sizeof( b_heap_t));
	b_h->h_arr = (int*)malloc( sizeof( int) * (l_max + 1 ));
	b_h->obj_arr = (h_obj_t*)malloc( sizeof( h_obj_t) * (l_max+1));
	b_h->size = l_max + 1;
	b_h->rear = 0;
		
	//read the data.
	for( i=1; i<l_max + 1; i++)
	{
		fscanf( i_fp, "%f", &(b_h->obj_arr[i].key));
	//	b_h->obj_arr[i].tag = 0;
	}

	//create a b_heap--test 'insert'.
	for( i=1; i<l_max+1; i++)
	{
		insert_b_heap( b_h, i);
	}

	//print the info. of the heap.
	fprintf( o_fp, "After creating the heap.\n");
	print_b_heap( o_fp, b_h);
	
	//test 'get-top'.
	top = get_top( b_h);
	fprintf( o_fp, "\n\n\nAfter getting the top node.\n");
	fprintf( o_fp, "The top: %f\n", b_h->obj_arr[top].key);

	//print the info. of the heap.
	print_b_heap( o_fp, b_h);

	//test 'update_key'.
	b_h->obj_arr[5].key = 0;
	update_key( b_h, 5);
	fprintf( o_fp, "\n\n\nAfter updating the key.\n");
	print_b_heap( o_fp, b_h);

	//release the related resources.
	fclose( c_fp);
	fclose( i_fp);
	fclose( o_fp);

	free( b_h->h_arr);
	free( b_h->obj_arr);
	free( b_h);
}
