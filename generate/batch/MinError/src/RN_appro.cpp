/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



#include "RN_appro.h"


/*
 *	Allocate a R_set_t structure.
 */
R_set_t* R_set_alloc( )
{
	R_set_t* R_set_v;

	R_set_v = ( R_set_t*)malloc( sizeof( R_set_t));
	memset( R_set_v, 0, sizeof( R_set_t));
	
	R_set_v->head = ( R_list_t*)malloc( sizeof( R_list_t));
	memset( R_set_v->head, 0, sizeof( R_list_t));

	/*s*/
	R_set_v->m_size += sizeof( R_set_t) + sizeof( R_list_t);
	/*s*/
	

	return R_set_v;
}

/*
 *	Release a R_set_t structure.
 */
void R_set_release( R_set_t* R_set_v)
{
	R_list_t* R_list_v1, *R_list_v2;

	/*s*/
	stat_v.memory_v -= R_set_v->m_size;
	/*s*/

	R_list_v1 = R_set_v->head;
	R_list_v2 = R_list_v1->next;
	while( R_list_v2 != NULL)
	{
		//R_node_ini( R_list_v1);
		free( R_list_v1);
		R_list_v1 = R_list_v2;
		R_list_v2 = R_list_v1->next;
	}
	//R_node_ini( R_list_v1);
	free( R_list_v1);
	free( R_set_v);
}

/*
 *	Construct a road (list) following a direction.
 *
 *	@RN_graph_v indicates the road network.
 *	@sta_id indicates the id of vertex from which the road starts.
 *	@g_node_v indicates the direction of the road from the vertex identified by sta_id.
 *
 *	@return the road stored in a R_list_t structure.
 */
R_list_t* R_list_const( RN_graph_t* RN_graph_v, int sta_id, RN_graph_list_t* g_node_v)
{
	int pre_sta_id;

	R_node_t* R_node_v1, *R_node_v2;
	R_list_t* R_list_v;
	
	
	R_list_v = R_list_alloc( );
	
	R_node_v1 = R_list_v->head;
	while( true)
	{
		//Construct node.
		R_node_v2 = ( R_node_t*)malloc( sizeof( R_node_t));
		memset( R_node_v2, 0, sizeof( R_node_t));

		/*s*/
		R_list_v->m_size += sizeof( R_node_t);
		/*s*/

		R_node_v2->id1 = sta_id;
		R_node_v2->id2 = g_node_v->id;
		R_node_v2->loc_v1 = &RN_graph_v->head_v[ sta_id].loc_v;	//Pay some attention here!
		R_node_v2->loc_v2 = &RN_graph_v->head_v[ g_node_v->id].loc_v;
		R_node_v2->leng = g_node_v->leng;

		//bug.
		//R_node_v2->slope = g_node_v->slope;
		R_node_v2->slope = calc_abs_slope( R_node_v2->loc_v1, R_node_v2->loc_v2);

		append_R_list_entry( R_list_v, R_node_v2);


		//Find the next edge (road segment).
		pre_sta_id = sta_id;
		sta_id = g_node_v->id;
		if( RN_graph_v->head_v[ sta_id].degree != 2)
		{
			//The other end of the road has been found.
			//Tag the corresponding edge such another identical road would not be generated.
			g_node_v = RN_graph_v->head_v[ sta_id].list->next;
			while( g_node_v != NULL)
			{
				if( g_node_v->id == pre_sta_id)
					break;

				g_node_v = g_node_v->next;
			}

			/*t*/
			if( g_node_v == NULL)
			{
				fprintf( stderr, "Incorrect graph informatin [R_list_const].\n");
				exit( 0);
			}
			/*t*/

			g_node_v->tag = 1;

			break;
		}

		g_node_v = RN_graph_v->head_v[ sta_id].list->next;
		if( g_node_v->id == pre_sta_id)		//Un-directed graph.
			g_node_v = g_node_v->next;

	}//while( true)

	return R_list_v;
}

/*
 *	Depcompose a road network into a set of roads.
 *
 *	@RN_graph_v indicates the road network.
 *
 *	@return the corresponding set of roads.
 */
R_set_t* RN_decompose( RN_graph_t* RN_graph_v)
{
	int i, vertex_n;

	R_list_t* R_list_v1, *R_list_v2;
	R_set_t* R_set_v;
	RN_graph_list_t* g_node_v;
	
	//Allocate the R_set_t structure for storing the decomposition result.
	R_set_v = R_set_alloc( );
	R_list_v1 = R_set_v->head;

	//Decomposition.
	vertex_n = RN_graph_v->vertex_n;
	for( i=0; i<vertex_n; i++)
	{
		//Find an appropriate starting point for searching.
		if( RN_graph_v->head_v[ i].degree == 2)
			continue;
		
		g_node_v = RN_graph_v->head_v[i].list->next;
		while( g_node_v != NULL)
		{
			//Traverse all possible "directions".
			if( g_node_v->tag == 0)
			{
				//Construct the corresponding road following this direction.
				R_list_v2 = R_list_const( RN_graph_v, i, g_node_v);
				R_list_v1->next = R_list_v2;
				R_list_v1 = R_list_v2;
				R_set_v->num ++;

				//bug.
				g_node_v->tag = 1;

				/*s*/
				R_set_v->m_size += R_list_v2->m_size;
				/*s*/
			}
			
			g_node_v = g_node_v->next;			
		}

	}//for( i)

	/*s*/
	stat_v.memory_v += R_set_v->m_size;
	if( stat_v.memory_max < stat_v.memory_v)
		stat_v.memory_max = stat_v.memory_v;
	/*s*/

	return R_set_v;
}

/*
 *	Decompose a slope range into two parts if necessary
 *	(the range is splitted by the positive x-axis).
 *
 *	@slope_r_v indicates the range to be decomposed.
 *	@slope_r_v1 and slope_r_v2 are used to store the decomposed results.
 */
void range_decompose( slope_r_t* slope_r_v, slope_r_t* & slope_r_v1, slope_r_t* & slope_r_v2)
{
	slope_r_v1 = ( slope_r_t*)malloc( sizeof( slope_r_t));
	slope_r_v2 = NULL;

	if( slope_r_v->s_end < slope_r_v->s_sta)
	{
		slope_r_v1->s_sta = 0;
		slope_r_v1->s_end = slope_r_v->s_end;

		slope_r_v2 = ( slope_r_t*)malloc( sizeof( slope_r_t));
		slope_r_v2->s_sta = slope_r_v->s_sta;
		slope_r_v2->s_end = 2 * PI;
	}
	else
	{
		slope_r_v1->s_sta = slope_r_v->s_sta;
		slope_r_v1->s_end = slope_r_v->s_end;
	}	
}

/*
 *	Calculate the intersection of two ranges.
 *
 *	@slope_r_v1 and slope_r_v2 indicate the two ranges, respectively.
 *	@slope_r_v is used to store the intersection information.
 
void range_intersect( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2, slope_r_t* & slope_r_v)
{
	float left, right;

	slope_r_v = NULL;

	/*t/
	if( slope_r_v1 == NULL || slope_r_v2 == NULL)
		return;
	/*t/
	
	left = slope_r_v1->s_sta > slope_r_v2->s_sta ? slope_r_v1->s_sta : slope_r_v2->s_sta;
	right = slope_r_v1->s_end < slope_r_v2->s_end ? slope_r_v1->s_end : slope_r_v2->s_end;

	if( left <= right)
	{
		slope_r_v = ( slope_r_t*)malloc( sizeof( slope_r_t));
		slope_r_v->s_sta = left;
		slope_r_v->s_end = right;
	}

	return;
}*/

/*
 *	Append the common range to the list.
 *
 *	@slope_r_v1 and slope_r_v2 indicate the two slope ranges, of which the common range should be appended.
 *	@slope_r_list_v indicates the list.
 
void append_common_range( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2, slope_r_list_t* slope_r_list_v)
{
	slope_r_t* slope_r_v;
	slope_r_list_t* slope_r_list_v1;

	range_intersect( slope_r_v1, slope_r_v2, slope_r_v);
	if( slope_r_v != NULL)
	{
		slope_r_list_v1 = ( slope_r_list_t*)malloc( sizeof( slope_r_list_t));
		//memset( slope_r_list_v1, 0, sizeof( slope_r_list_t));
		slope_r_list_v1->slope_r_v = slope_r_v;
		slope_r_list_v1->next = NULL;
		slope_r_list_v->next = slope_r_list_v1;
	}
}
*/

/*
 *	Calculate the common range between two given slope ranges.
 *	
 *	@slope_r_v1 and slope_r_v2 indicate the two given ranges.
 *	
 *	@return the common slope range.
 
slope_r_list_t* calc_common_range( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2)
{
	slope_r_t* slope_r_v11, *slope_r_v12, *slope_r_v21, *slope_r_v22;
	slope_r_list_t* slope_r_list_v;

	slope_r_list_v = ( slope_r_list_t*)malloc( sizeof( slope_r_list_t));
	memset( slope_r_list_v, 0, sizeof( slope_r_list_t));

	range_decompose( slope_r_v1, slope_r_v11, slope_r_v12);
	range_decompose( slope_r_v2, slope_r_v21, slope_r_v22);
	
	if( slope_r_v11 != NULL && slope_r_v21 != NULL)
		append_common_range( slope_r_v11, slope_r_v21, slope_r_list_v);

	if( slope_r_v11 != NULL && slope_r_v22 != NULL)
		append_common_range( slope_r_v11, slope_r_v21, slope_r_list_v);

	if( slope_r_v12 != NULL && slope_r_v21 != NULL)
		append_common_range( slope_r_v11, slope_r_v21, slope_r_list_v);

	if( slope_r_v12 != NULL && slope_r_v22 != NULL)
		append_common_range( slope_r_v11, slope_r_v21, slope_r_list_v);

	if( slope_r_v11 != NULL)
		free( slope_r_v11);
	if( slope_r_v12 != NULL)
		free( slope_r_v12);
	if( slope_r_v21 != NULL)
		free( slope_r_v21);
	if( slope_r_v22 != NULL)
		free( slope_r_v22);

	return slope_r_list_v;
}*/

/*
 *	Calculate the angle between two vectors.
 *
 *	@(loc_v11, loc_v12) indicates vector 1.
 *	@(loc_v21, loc_v22) indicates vector 2.
 *
 *	@return the angle between the two vectors.
 */
float calc_angle( loc_t* loc_v11, loc_t* loc_v12, loc_t* loc_v21, loc_t* loc_v22)
{
	float v11, v12, v21, v22, leng1, leng2, dot_product, cos_v, angle_v;

	v11 = loc_v12->x - loc_v11->x;
	v12 = loc_v12->y - loc_v11->y;

	v21 = loc_v22->x - loc_v21->x;
	v22 = loc_v22->y - loc_v21->y;

	leng1 = pow( v11, 2) + pow( v12, 2);
	leng2 = pow( v21, 2) + pow( v22, 2);


	//Don't count those edges with small lengths.
	//if( leng1 > -1e-6 && leng1 < 1e-6 || 
	//	leng2 > -1e-6 && leng2 < 1e-6)
	//	return 0;


	leng1 = sqrt( leng1);
	leng2 = sqrt( leng2);

	dot_product = v11 * v21 + v12 * v22;

	cos_v = dot_product / ( leng1 * leng2);

	if( cos_v == FLT_MAX)
		return 0;

	//if( cos_v == INF

	//if( errno == EDOM)
	//	printf( "dot_product: %f\nleng1: %f\nleng2: %f\n\n", dot_product, leng1, leng2);

	if( !( cos_v >= -1 && cos_v <= 1))
		return 0;

	/*special handling*/
	if( cos_v == 1)
		return 0;
	else
		angle_v = ( float)acos( cos_v);
	
	/*special handling*/
	if( angle_v < 1e-6 && angle_v > -1e-6)
		return 0;

	//if( angle_v == nan)
	//	return 0;

	return angle_v;
}

/*
 *	Calculate the angular difference between two directions.
 */
float calc_angular_diff( float slope_v1, float slope_v2)
{
	float abs_v;

	abs_v = abs_f( slope_v1 - slope_v2); 

	return abs_v < 2 * PI - abs_v ? abs_v : 2 * PI - abs_v;
}

/*
 *	Calculate the error if all the road segments between two road segments are approximated by one.
 */
float calc_error( R_node_t* R_node_v1, R_node_t* R_node_v2)
{	
	float angle_v, max_angle, slope_v;
	R_node_t* R_node_v3, *R_node_v4;

	max_angle = 0;

	//t
	if( R_node_v1->id1 == 71 && R_node_v2->id1 == 72)
		printf( "");
	//t

	slope_v = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);

	R_node_v3 = R_node_v2->next;
	R_node_v4 = R_node_v1;
	while( R_node_v4 != R_node_v3)
	{
		//angle_v = calc_angle( R_node_v4->loc_v1, R_node_v4->loc_v2,
		//	R_node_v1->loc_v1, R_node_v2->loc_v2);
		angle_v = calc_angular_diff( R_node_v4->slope, slope_v);

		if( angle_v > max_angle)
		{
			max_angle = angle_v;
		}

		R_node_v4 = R_node_v4->next;
	}

	return max_angle;
}


/*
 *	Calculate the error if all the road segments between two road segments are approximated by one.
 *
 *	@R_nodoe_v1 and R_node_v2 indicate the beginning and end road segments, respectively.
 *	
 *	@return the approximation error.
 */
int check_error( R_node_t* R_node_v1, R_node_t* R_node_v2, float eps)
{
	float angle_v, max_angle;
	R_node_t* R_node_v3, *R_node_v4;

	max_angle = 0;

	R_node_v3 = R_node_v2->next;
	R_node_v4 = R_node_v1;
	while( R_node_v4 != R_node_v3)
	{
		angle_v = calc_angle( R_node_v4->loc_v1, R_node_v4->loc_v2,
			R_node_v1->loc_v1, R_node_v2->loc_v2);

		if( angle_v > max_angle)
		{
			max_angle = angle_v;

			if( max_angle > eps)
				return 0;
		}

		R_node_v4 = R_node_v4->next;
	}

	return 1;
}

/*
 *	Calculate the length information of a road segment.
 *
 *	@loc_v1 and loc_v2 indicate the two ends of the road segment.
 *
 *	@return the length of the segment.
 */
float calc_leng( loc_t* loc_v1, loc_t* loc_v2)
{	
	return (float)sqrt( pow( loc_v2->x - loc_v1->x, 2) + 
					pow( loc_v2->y - loc_v1->y, 2));
}

//Min-Number.


/*
 *	A straight-forward implementation of "cal_appr_table", running in O(n^3).
 *	The space cost is O(n^2) since it maintains a matrix/table.
 *	In fact, the space cost could be reduced to O(n) by not materilizing the table.
 *	
 *	@appr[i][j] indicates whether \epsilon( \overline{ v_{i}v_{j+1}}) is at most eps.
 */
int** calc_appr_table1( R_list_t* R_list_v, float eps)
{
	int i, j, seg_n;
	int** appr;

	R_node_t* R_node_v1, *R_node_v2;

	//Allocate the 2^d array for storing the information.
	seg_n = R_list_v->size;

	appr = ( int**)malloc( sizeof( int*) * seg_n);
	memset( appr, 0, sizeof( int*) * seg_n);

	i = 0;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1 != NULL)
	{
		appr[ i] = ( int*)malloc( sizeof( int) * seg_n);
		memset( appr[ i], 0, sizeof( int) * seg_n);

		appr[ i][ i] = 1;

		//R_node_v1 is the starting segment.
		//R_node_v2 is the ending segment.
		R_node_v2 = R_node_v1->next;
		j = i + 1;
		while( R_node_v2 != NULL)
		{
			//Compute the error of approximating the segments between R_node_v1 and R_node_v2 with one segment.
			appr[ i][ j] =  check_error( R_node_v1, R_node_v2, eps);

			R_node_v2 = R_node_v2->next;
			j++;
		}//

		R_node_v1 = R_node_v1->next;
		i++;
	}//


	/*s*/
	emp_stat_v.memory_v += sizeof( int*) * seg_n + sizeof( int) * seg_n * seg_n;
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return appr;
}

/*
 *	Solve the Min-Number problem for a road using the DP algorithm.
 *
 *	@R_list_v indicates the road.
 *	@eps is the error tolerance.
 *	@ver_opt indicates the version of the algorithm.
 *		= 0: straightforward method of computing \epsilon.
 *		= 1: with the help of the fdr information.
 *	
 *	@return the minimized number of line segments.
 */
int DP_Number( R_list_t* R_list_v, float eps, int ver_opt)
{
	int i, j, k, seg_n, min, rtn;
	int** PN, **appr_table;

	//Initialize the problem space.
	seg_n = R_list_v->size;

	PN = ( int**)malloc( sizeof( int*) * seg_n);
	memset( PN, 0, sizeof( int*) * seg_n);

	for( i=0; i<seg_n; i++)
	{
		PN[ i] = ( int*)malloc( sizeof( int) * seg_n);
		memset( PN[ i], 0, sizeof( int) * seg_n);

		PN[ i][ i] = 1;
	}

	/*s*/
	emp_stat_v.memory_v += sizeof( int*) * seg_n + sizeof( int) * seg_n * seg_n;
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	//Collect the appr_table information.
	if( ver_opt == 0)
		appr_table = calc_appr_table1( R_list_v, eps);
	else //if (ver_opt == 1)
		appr_table = calc_appr_table2( R_list_v, eps);

	/*t/
	print_appr_table( appr_table, R_list_v->size, R_list_v->size);
	/*t*/

	//Solve all the problem instances.
	for( j=1; j<seg_n; j++)
	{
		for( i=0; i<seg_n-j; i++)
		{
			if( appr_table[ i][ i+j] == 1)
				PN[ i][ i+j] = 1;

			else
			{
				min = INT_MAX;
				for( k=i; k<i+j; k++)
				{
					if( PN[ i][ k] + PN[ k+1][ i+j] < min)
						min = PN[ i][ k] + PN[ k+1][ i+j];
				}
				
				PN[ i][ i+j] = min;
			}
		}//for( i)
	}//for( j)

	rtn = PN[ 0][ seg_n-1];

	//Release the related resources.
	for( i=0; i<seg_n; i++)
	{
		free( appr_table[ i]);
		free( PN[ i]);
	}

	free( appr_table);
	free( PN);

	/*s*/
	emp_stat_v.memory_v -= ( sizeof( int*) * seg_n + sizeof( int) * seg_n * seg_n) * 2;
	/*s*/

	return rtn;
}

/*
 *	Allocate the eps_g_t structure.
 *
 *	@vertex_n indicates the number of vertices.
 */
eps_g_t* eps_g_alloc( int vertex_n)
{
	int i;
	eps_g_t* eps_g_v;
	eps_g_node_t* g_node_v;

	eps_g_v = ( eps_g_t*)malloc( sizeof( eps_g_t));
	memset(  eps_g_v, 0, sizeof( eps_g_t));

	eps_g_v->vertex_n = vertex_n;
	eps_g_v->head = ( eps_g_head_t*)malloc( sizeof( eps_g_head_t) * vertex_n);
	
	for( i=0; i<vertex_n; i++)
	{
		g_node_v = ( eps_g_node_t*)malloc( sizeof( eps_g_node_t));
		memset( g_node_v, 0, sizeof( eps_g_node_t));

		eps_g_v->head[ i].list = g_node_v;
	}

	/*s*/
	emp_stat_v.memory_v += sizeof( eps_g_t) +
							sizeof( eps_g_head_t) * vertex_n +
							sizeof( eps_g_node_t) * vertex_n;;
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return eps_g_v;
}

/*
 *	Release the eps_g_t structure.
 *	
 *	@eps_g_v indicates the eps_g_t structure.
 */
void eps_g_release( eps_g_t* eps_g_v)
{
	int i;
	eps_g_node_t* g_node_v1, *g_node_v2;

	for( i=0; i<eps_g_v->vertex_n; i++)
	{
		g_node_v1 = eps_g_v->head[ i].list;


		while( g_node_v1 != NULL)
		{

			g_node_v2 = g_node_v1->next;

			free( g_node_v1);

			/*s*/
			emp_stat_v.memory_v -= sizeof( eps_g_node_t);
			/*s*/

			g_node_v1 = g_node_v2;
		}
	}

	/*s*/
	emp_stat_v.memory_v -= eps_g_v->vertex_n * sizeof( eps_g_head_t) + sizeof( eps_g_t);
	/*s*/
	
	free( eps_g_v->head);
	free( eps_g_v);	
}

/*
 *	Construct the "G_{\epsilon_t}".
 * 
 *	@R_list_v indicates the road (list).
 *	@eps is the error tolerance.
 *
 *	@return the "G_{\epsilon_t}".
 */
eps_g_t* eps_g_const( R_list_t* R_list_v, float eps, int opt)
{
	int i, j, seg_n, vertex_n;
	int** appr_table;

	eps_g_t* eps_g_v;
	eps_g_node_t* g_node_v1, *g_node_v2;
	R_node_t* R_node_v1, *R_node_v2, *R_node_v3;

	//Allocate a eps_g_t structure.
	seg_n = R_list_v->size;
	vertex_n = seg_n + 1;
	eps_g_v = eps_g_alloc( vertex_n);

	//Collect the appr_table information.
	//appr_table = calc_appr_table1( R_list_v, eps);

	/*t/
	printf( "The implementation with cost O(n^3):\n");
	print_appr_table( appr_table, seg_n, seg_n);
	/*t*/
	printf( "Computing the error table ...\n");
	if( opt == 0)
		appr_table = calc_appr_table1( R_list_v, eps);
	else
		appr_table = calc_appr_table2( R_list_v, eps);

	/*t/
	print_appr_table( appr_table, seg_n, seg_n);
	/*t*/

	//Construct the "G_{\epsilon_t}".
	R_node_v3 = NULL;
	R_node_v1 = R_list_v->head->next;

	for( i=0; i<seg_n; i++)
	{
		//eps_g_v->head[ i].ori_id = R_node_v1->id1;

		g_node_v1 = eps_g_v->head[ i].list;

		R_node_v2 = R_node_v1;
		for( j=i+1; j<vertex_n; j++)
		{
			if( appr_table[ i][ j-1] == 1)
			{
				g_node_v2 = ( eps_g_node_t*)malloc( sizeof( eps_g_node_t));
				memset( g_node_v2, 0, sizeof( eps_g_node_t));

				g_node_v2->id = j;
				//g_node_v2->ori_id = R_node_v2->id2;
			
				g_node_v1->next = g_node_v2;
				g_node_v1 = g_node_v2;

				/*s*/
				emp_stat_v.memory_v += sizeof( eps_g_node_t);
				/*s*/
			}

			R_node_v2 = R_node_v2->next;
		}

		R_node_v3 = R_node_v1;
		R_node_v1 = R_node_v1->next;
	}//for( i)

	//eps_g_v->head[ seg_n].ori_id = R_node_v3->id2;

	/*s*/
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	//Release the related resources.
	for( i=0; i<seg_n; i++)
		free( appr_table[ i]);
	free( appr_table);

	/*s*/
	emp_stat_v.memory_v -= sizeof( int*) * seg_n + sizeof( int) * seg_n * seg_n;
	/*s*/

	return eps_g_v;
}

/*
 *	The implementation of eps_g_const without the maintenance of the 2D array.
 *	The straightforward method is used to compute the error.
 */
eps_g_t* eps_g_const1( R_list_t* R_list_v, float eps)
{
	int i, j, seg_n, vertex_n;
	//int** appr;

	R_node_t* R_node_v1, *R_node_v2;

	eps_g_t* eps_g_v;
	eps_g_node_t* eps_g_node_v;

	//Allocate the 2^d array for storing the information.
	seg_n = R_list_v->size;
	vertex_n = seg_n + 1;

	eps_g_v = eps_g_alloc( vertex_n);

	i = 0;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1 != NULL)
	{
		//appr[ i][ i] = 1;
		eps_g_node_v = ( eps_g_node_t*)malloc( sizeof( eps_g_node_t));
		memset( eps_g_node_v, 0, sizeof( eps_g_node_t));

		eps_g_node_v->id = R_node_v1->id2;
		eps_g_node_v->next = eps_g_v->head[ i].list->next;
		eps_g_v->head[ i].list->next = eps_g_node_v;
		eps_g_v->edge_n ++;	

		//R_node_v1 is the starting segment.
		//R_node_v2 is the ending segment.
		R_node_v2 = R_node_v1->next;
		j = i + 1;
		while( R_node_v2 != NULL)
		{
			//Compute the error of approximating the segments between R_node_v1 and R_node_v2 with one segment.
			if( check_error( R_node_v1, R_node_v2, eps))
			{
				//Add an edge in the eps_g_v;
					eps_g_node_v = ( eps_g_node_t*)malloc( sizeof( eps_g_node_t));
					memset( eps_g_node_v, 0, sizeof( eps_g_node_t));
					
					eps_g_node_v->id = R_node_v2->id2;
					eps_g_node_v->next = eps_g_v->head[ i].list->next;
					eps_g_v->head[ i].list->next = eps_g_node_v;
					eps_g_v->edge_n ++;	
			}

			R_node_v2 = R_node_v2->next;
			j++;

			/*s*/
			emp_stat_v.eps_n ++;
			/*s*/
		}//

		R_node_v1 = R_node_v1->next;
		i++;
	}//


	/*s*/
	emp_stat_v.memory_v += eps_g_v->edge_n * sizeof( eps_g_node_t);
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return eps_g_v;
}

/*
 *	The implementation of eps_g_const without the maintenance of the 2D array.
 *	The error is checked with the help of the fdr information.
 */
eps_g_t* eps_g_const2( R_list_t* R_list_v, float eps)
{
	int i, j, seg_n, vertex_n;
	//int** error_tab;
	float slope_v, slope_v1;

	slope_r_list_t* slope_r_list_v1, *slope_r_list_v2;
	slope_r_t slope_r_v;

	R_node_t* R_node_v1, *R_node_v2;

	eps_g_t* eps_g_v;
	eps_g_node_t* eps_g_node_v;


	seg_n = R_list_v->size;
	vertex_n = seg_n + 1;

	eps_g_v = eps_g_alloc( vertex_n);


	R_node_v1 = R_list_v->head->next;
	//for( i=0; i<seg_n; i++)
	i = 0;
	while( R_node_v1 != NULL)
	{
		//error_tab[ i] = ( int*)malloc( seg_n * sizeof( int));
		//memset( error_tab[ i], 0, seg_n * sizeof( int));

		//The fdr of the first segment.
		slope_v = R_node_v1->slope;
		slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
		slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

		//Maintain the slope range list.
		slope_r_list_v1 = slope_r_list_alloc( );
		append_slope_r_list_entry( slope_r_list_v1, slope_r_v);

		
		//error_tab[ i][ i] = 1;
		eps_g_node_v = ( eps_g_node_t*)malloc( sizeof( eps_g_node_t));
		memset( eps_g_node_v, 0, sizeof( eps_g_node_t));

		eps_g_node_v->id = R_node_v1->id2;
		eps_g_node_v->next = eps_g_v->head[ i].list->next;
		eps_g_v->head[ i].list->next = eps_g_node_v;
		eps_g_v->edge_n ++;	

		/*s*/
		emp_stat_v.eps_n ++;
		/*s*/
		

		R_node_v2 = R_node_v1->next;
		//for( j=i+1; j<seg_n; j++)
		j = i + 1;
		while( R_node_v2 != NULL)
		{
			slope_v = R_node_v2->slope;
			slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
			slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

			slope_r_list_v2 = range_intersect( slope_r_list_v1, &slope_r_v);

			slope_r_list_release( slope_r_list_v1);
			slope_r_list_v1 = NULL;

			/*s*/
			emp_stat_v.eps_n ++;
			/*s*/

			if( slope_r_list_v2 == NULL)
				break;
			else	//slope_r_list_v2 != NULL.
			{
				//bug.
				slope_v1 = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);
				if( is_in_slope_r_list( slope_r_list_v2, slope_v1))
				{
					//error_tab[ i][ j] = 1;

					//Add an edge in the eps_g_v;
					eps_g_node_v = ( eps_g_node_t*)malloc( sizeof( eps_g_node_t));
					memset( eps_g_node_v, 0, sizeof( eps_g_node_t));
					
					eps_g_node_v->id = R_node_v2->id2;
					eps_g_node_v->next = eps_g_v->head[ i].list->next;
					eps_g_v->head[ i].list->next = eps_g_node_v;
					eps_g_v->edge_n ++;	
				}

				slope_r_list_v1 = slope_r_list_v2;
			}

			R_node_v2 = R_node_v2->next;
			j++;
		}//

		slope_r_list_release( slope_r_list_v1);
		R_node_v1 = R_node_v1->next;
		i++;
	}//

	/*s*/
	emp_stat_v.memory_v += eps_g_v->edge_n * sizeof( eps_g_node_t);
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return eps_g_v;
}

/*
 *	Calculate the length of the shortest path from s_id to d_id.
 *
 *	@eps_g_v indicates the graph.
 *	@s_id and d_id indicate the source and destination, repectively.
 *	
 *	@return the length of the shortest path.
 */
int calc_SP1( eps_g_t* eps_g_v, int s_id, int d_id)
{
	int i, vertex_n, sta, rear, c_id, n_id, rtn;
	int* que, *level, *pre;
	eps_g_node_t* g_node_v;

	vertex_n = eps_g_v->vertex_n;
	que = ( int*)malloc( sizeof( int) * vertex_n);
	level = ( int*)malloc( sizeof( int) * vertex_n);
	pre = ( int*)malloc( sizeof( int) * vertex_n);
	memset( que, 0, sizeof( int) * vertex_n);
	memset( level, 0, sizeof( int) * vertex_n);
	memset( pre, 0, sizeof( int) * vertex_n);

	/*s*/
	emp_stat_v.memory_v += sizeof( int) * vertex_n * 3;
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/
	
	for( i=0; i<vertex_n; i++)
		pre[ i] = -2;

	rtn = 0;

	sta = rear = -1;
	que[ ++rear] = s_id;
	pre[ s_id] = -1; 
	while( sta < rear)
	{
		c_id = que[ ++sta];

		g_node_v = eps_g_v->head[ c_id].list->next;
		while( g_node_v != NULL)
		{
			n_id = g_node_v->id;
			if( n_id == d_id)
			{
				rtn = level[ c_id] + 1;
				break;
			}

			if( pre[ n_id] == -2)
			{
				/*t*/
				if( rear == vertex_n - 1)
				{
					fprintf( stderr, "Incorrect eps_g_v [calc_SP].\n");
					exit( 0);
				}
				/*t*/

				que[ ++rear] = n_id;
				level[ n_id] = level[ c_id] + 1;
				pre[ n_id] = c_id;
			}
			
			g_node_v = g_node_v->next;
		}//while( g_node_v)

		if( rtn != 0)
			break;
	}//while( sta < rear)

	free( que);
	free( level);
	free( pre);
	
	/*s*/
	emp_stat_v.memory_v -= sizeof( int) * vertex_n * 3;
	/*s*/

	return rtn;
}

/*
 *	Solve the Min-Number problem for a road using SP-Method.
 *
 *	@R_list_v and @eps are the the same as in DP_Number.
 *	@ver_opt indicates the version of SP_Method.
 *		ver_opt = 0: SP-Method-B.
 *		ver_opt = 1: SP-Method-E1.
 *
 */
int SP_Method_sub( R_list_t* R_list_v, float eps, int ver_opt)
{
	int rtn, seg_n;
	eps_g_t* eps_g_v;

	seg_n = R_list_v->size;

	//Construct the "G_{\epsilon_t}" graph.
	printf( "Constructing the graph ...\n");
	//eps_g_v = eps_g_const( R_list_v, eps, ver_opt);

	if( ver_opt == 0)
		eps_g_v = eps_g_const1( R_list_v, eps);
	else
		eps_g_v = eps_g_const2( R_list_v, eps);

	/*t/
	printf( "G_{eps=%f}:\n", eps);
	print_eps_g( eps_g_v);
	/*t*/

	//Calculate the shortest path from vertex identified by 0 to that identified by "seg_n".
	printf( "Computing the shortest path ...\n");
	rtn = calc_SP1( eps_g_v, 0, seg_n);

	//Release the related resources.
	eps_g_release( eps_g_v);

	return rtn;
}

/*
 *	Sub-component of Min_Number.
 *
 *	opt:
 *		= 1: DP;
 *		= 2: SP-Method-B
 *		= 3: SP-Method-E1
 *		= 4: SP-Method-E2
 */
int Min_Number_sub( R_set_t* R_set_v, float eps, int opt)
{
	int sum;
	R_list_t* R_list_v;

	/*t/
	printf( "seg_after the decomposition:\n");
	print_R_set( R_set_v);
	/*t*/

	//Approximate the roads separately.
	sum = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		if( opt == 1)
			sum += DP_Number( R_list_v, eps, 1);
		else if( opt == 2)
			sum += SP_Method_sub( R_list_v, eps, 1);
		else if( opt == 3)
			sum += SP_Method_sub( R_list_v, eps, 2);
		else if( opt == 4)
			//

		R_list_v = R_list_v->next;
	}
	
	return sum;
}

/*
 *	Solve the Min-Number problem for a road network.
 *
 *	@RN_graph_v indicates the road network.
 *	@eps is the error tolerance.
 *	@opt indicates the choice of methods.
 *		--opt = 1: SP-Method.
 *		--opt = 2: DP-Number.
 *
 *	@return the minimized the number of road segments.
 */
int Min_Number( RN_graph_t* RN_graph_v, float eps, int opt)
{
	int num;
	R_set_t* R_set_v;

	//Decompose the road network into a set of roads.
	R_set_v = RN_decompose( RN_graph_v);

	//
	num = Min_Number_sub( R_set_v, eps, opt);

	//Release the related resources.
	R_set_release( R_set_v);

	return num;
}


//Min-Error.

/*
 *	Sorting use.
 */
int compare (const void * a, const void * b)
{
	return *( float*)a > *( float*)b ? 1 : ( *( float*)a == *( float*)b ? 0 : -1);
}

/*
 *	Construct the search space for "Search_Error_R" [O(n^3) implementation].
 *
 *	@R_list_v indicates the road.
 *
 *	@return the search space [array].
 */
float* search_space_R( R_list_t* R_list_v)
{
	int i, j, seg_n, s_size, cnt;
	float* s_space;
	R_node_t* R_node_v1, *R_node_v2;

	seg_n = R_list_v->size;
	s_size = seg_n * ( seg_n - 1) / 2;
	s_space = ( float*)malloc( sizeof( float) * s_size);
	memset( s_space, 0, sizeof( float) * s_size);

	cnt = 0;
	R_node_v1 = R_list_v->head->next;
	for( i=0; i<seg_n-1; i++)
	{
		R_node_v2 = R_node_v1;
		for( j=i+1; j<seg_n; j++)
		{
			R_node_v2 = R_node_v2->next;

			s_space[ cnt++] =  calc_error( R_node_v1, R_node_v2);
			
			/*t*/
			if( cnt > s_size)
			{
				fprintf( stderr, "s_space array overflow [search_space_R].\n");
				exit( 0);
			}
			/*t*/
		}//for( j)

		R_node_v1 = R_node_v1->next;
	}//for( i)
			
	/*t*/
	if( cnt != s_size)
	{
		fprintf( stderr, "Incorrect s_space information [search_space_R].\n");
		exit( 0);
	}
	/*t*/

	return s_space;
}

/*
 *	Solve the Min-Error problem on a road using the Search-Error-Road algorithm.
 *
 *	@R_list_v indicates the road.
 *	@L indicates the complexity requirement (quota).
 *
 *	@return the minimized error.
 */
float Search_Error_R( R_list_t* R_list_v, int L)
{
	int seg_n, s_size, left, right, mid, res;
	float rtn;
	float* s_space;

	seg_n = R_list_v->size;
	s_size = seg_n * ( seg_n - 1) / 2 + 1;

	if( s_size == 0)
		return 0;

	//Construct the search space.
	s_space = search_space_R3( R_list_v);

	/*t/
	int i;
	printf( "search space (seg_before sorting):\n");
	for( i=0; i<s_size; i++)
		printf( "%f  ", s_space[ i]);
	printf( "\n\n");
	/*t*/

	//Sort the elements in the search space.
	qsort( s_space, s_size, sizeof( float), compare);

	/*t/
	int i;
	printf( "search space (seg_after sorting):\n");
	for( i=0; i<s_size; i++)
		printf( "%f  ", s_space[ i]);
	printf( "\n\n");
	/*t*/

	//Bisection method.
	left = 0;
	right = s_size - 1;
	while( left < right)
	{
		//Invariate: s_sapce[ right] must be feasible while s_space[ left-1] must not be feasible.
		mid = ( left + right) / 2;

		res = SP_Method_sub( R_list_v, s_space[ mid], 2);
		if( res <= L)
			right = mid;
		else
			left = mid + 1;
	}

	rtn = s_space[ right];

	//Release the related resources.
	free( s_space);

	return rtn;
}

/* 
 *	Used as a sub-component for the DP_Error_R function.
 *
 *	@R_node_v1 and R_node_v2 indicate the beginning and end road segments, respectively.
 *	@L indicates the "complexity requirement" (quota).
 *
 *	@return the minimized error.
 */
float DP_Error_R_sub( R_node_t* R_node_v1, R_node_t* R_node_v2, int L)
{
	int l;
	float min_error, error_v, error_v1, error_v2;
	R_node_t* R_node_v3;

	//Base problem instance 1.
	if( R_node_v1 == R_node_v2)
		return 0;

	//Base problem instance 2.
	if( L == 1)
		return calc_error( R_node_v1, R_node_v2);

	//Other problem instances.
	min_error = FLT_MAX;
	R_node_v3 = R_node_v1;
	while( R_node_v3 != R_node_v2)
	{
		for( l=1; l<L; l++)
		{
			error_v1 = DP_Error_R_sub( R_node_v1, R_node_v3, l);
			error_v2 = DP_Error_R_sub( R_node_v3->next, R_node_v2, L-l);

			error_v = error_v1 > error_v2 ? error_v1 : error_v2;

			if( min_error > error_v)
				min_error = error_v;

		}//for( l)

		R_node_v3 = R_node_v3->next;
	}//while( R_node_v3)

	return min_error;
}

/*
 *	Solve the Min-Error problem on a road using the DP-Error-Road algorithm.
 *
 *	@Same interface as Search_Error_R.
 */
float DP_Error_R( R_list_t* R_list_v, int L)
{
	R_node_t* R_node_v1, *R_node_v2;

	//Locate the road segments.
	R_node_v1 = R_list_v->head->next;

	R_node_v2 = R_node_v1;
	while( R_node_v2->next != NULL)
		R_node_v2 = R_node_v2->next;

	//Invoke the recursive function.
	return DP_Error_R_sub( R_node_v1, R_node_v2, L);
}

/*
 *	Construct the search space for Search_Error_Network.
 *
 *	@RN_graph_v indicates the road network.
 *
 *	@return the search space information.
 */
search_space_t* search_space_N( RN_graph_t* RN_graph_v)
{
	int i, s_size, cnt, seg_n, s_size_tmp;
	float* s_space, *s_space_tmp;
	search_space_t* search_space_v;
	R_set_t* R_set_v;
	R_list_t* R_list_v;

	search_space_v = ( search_space_t*)malloc( sizeof( search_space_t));
	memset( search_space_v, 0, sizeof( search_space_t));

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);
	
	//Collect the size information of the search space.
	cnt = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		cnt += seg_n * ( seg_n - 1) / 2;

		R_list_v = R_list_v->next;
	}

	//Allocate the memory.
	s_size = cnt;
	s_space = ( float*)malloc( sizeof( float) * s_size);
	memset( s_space, 0, sizeof( float) * s_size);

	//Collect the possible error information.
	cnt = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		s_size_tmp = seg_n * ( seg_n - 1) / 2;
		s_space_tmp = search_space_R( R_list_v);
		for( i=0; i<s_size_tmp; i++)
			s_space[ cnt++] = s_space_tmp[ i];

		R_list_v = R_list_v->next;

		free( s_space_tmp);
	}

	search_space_v->s_space = s_space;
	search_space_v->size = s_size;

	//Release the related resources.
	R_set_release( R_set_v);

	return search_space_v;
}

/*
 *	Solve the Min-Error on a road network using the Search_Error_Network algorithm.
 *
 *	@RN_graph_v indicates the road network.
 *	@L indicates the "complexity requirement" (quota).
 *	@opt indicates the method for "feasibility testing".
 *		--opt = 1: SP-Method.
 *		--opt = 2: DP-Number.
 *
 *	@return the minimized error.
 */
float Search_Error_N( RN_graph_t* RN_graph_v, int L, int opt)
{
	int left, mid, right;
	float res;
	float* s_space;
	
	search_space_t* search_space_v;
	R_set_t* R_set_v;

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);

	/*tr/
	print_R_set( R_set_v);
	/*t*/

	//Construct the search space information.
	search_space_v = search_space_N3( R_set_v);

	/*t/
	printf( "Search space (seg_before sorting):\n");
	print_search_space( search_space_v);
	/*t*/

	//Sort the elements in the search space.
	qsort( search_space_v->s_space, search_space_v->size, sizeof( float), compare);

	/*t/
	printf( "Search space (seg_after sorting):\n");
	print_search_space( search_space_v);
	/*t*/

	//Perform the bisection process.
	left = 0;
	s_space = search_space_v->s_space;
	right = search_space_v->size - 1;
	while( left < right)
	{
		//Invariate: s_space[ right] must be feasible while s_space[ left-1] must not be feasible.
		mid = ( left + right) / 2;

		if( Min_Number_sub( R_set_v, s_space[ mid], opt) <= L)
			right = mid;
		else
			left = mid + 1;
	}

	res = s_space[ right];

	//Release the related resources.

	/*s*/
	stat_v.memory_v -= sizeof( float) * search_space_v->size + 
						sizeof( search_space_t);
	/*s*/

	free( search_space_v->s_space);
	free( search_space_v);
	R_set_release( R_set_v);

	return res;
}

/*
 *	Used as a sub component of DP_Error_N.
 *
 *	@Solve the Min-Errro problem on a subset of roads of the road network.
 *	@This subset of roads are identified from R_list_v1 to R_list_v2.
 *	@num indicates the number of the roads in this subset.
 *	@L indicates the "complexity requirement" (quota).
 *	@opt indicates the method for solving the Min-Error problem on a road.
 *		--opt = 1: Search_Error_R.
 *		--opt = 2: DP_Error_R.
 *
 *	@return the minimized error.
 */
float DP_Error_N_sub( R_list_t* R_list_v1, R_list_t* R_list_v2, int num, int L, int opt)
{
	//1. L should be at least the number of roads in this subset of roads.

	int l, k;
	float max_error, min_error, error_v, error_v1, error_v2;
	R_list_t* R_list_v3;

	//Base problem instance 1.
	if( R_list_v1 == R_list_v2)
	{
		if( opt == 1)
			return Search_Error_R( R_list_v1, L);
		else
			return DP_Error_R( R_list_v1, L);
	}

	//Base problem instance 2.
	if( L == num)
	{
		R_node_t* R_node_v1, *R_node_v2;

		max_error = 0;
		R_list_v3 = R_list_v2->next;
		while( R_list_v1 != R_list_v3)
		{
			/*
			if( opt == 1)
				error_v = Search_Error_R( R_list_v1, 1);
			else
				error_v = DP_Error_R( R_list_v1, 1);
			*/
			R_node_v1 = R_list_v1->head->next;
			R_node_v2 = R_node_v1;
			while( R_node_v2->next != NULL)
				R_node_v2 = R_node_v2->next;
			error_v = calc_error(  R_node_v1, R_node_v2);

			if( max_error < error_v)
				max_error = error_v;

			R_list_v1 = R_list_v1->next;
		}//while( R_list_v1)

		return max_error;
	}

	//Other problem instances.
	min_error = FLT_MAX;

	k = 1;
	R_list_v3 = R_list_v1;
	while( R_list_v3 != R_list_v2)
	{
		// k <= L - ( num -k)
		for( l=k; l<=L-(num-k); l++)
		{
			error_v1 = DP_Error_N_sub( R_list_v1, R_list_v3, k, l, opt);
			error_v2 = DP_Error_N_sub( R_list_v3->next, R_list_v2, num-k, L-l, opt);

			error_v = max( error_v1, error_v2);

			if( error_v < min_error)
				min_error = error_v;
		}

		R_list_v3 = R_list_v3->next;
		k++;
	}
	
	/*t/
	printf( "num:  %i\n", num);
	printf( "L:  %i\n\n", L);
	/*t*/

	return min_error;
}

/*
 *	Solve the Min-Error problem on a road network using a DP algorithm.
 *
 *	@Same interface as Search_Error_N.
 */
float DP_Error_N( RN_graph_t* RN_graph_v, int L, int opt)
{
	float res;
	R_set_t* R_set_v;
	R_list_t* R_list_v1, *R_list_v2;

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);

	/*t/
	printf( "seg_after the decomposition:\n");
	print_R_set( R_set_v);
	/*t*/

	//Pre-check.
	if( L < R_set_v->num)
	{
		fprintf( stderr, "Insufficient quota [DP_Error_N].\n");
		
		R_set_release( R_set_v);
		return FLT_MAX;
	}

	//Locate the roads.
	R_list_v1 = R_set_v->head->next;
	R_list_v2 = R_list_v1;
	while( R_list_v2->next != NULL)
		R_list_v2 = R_list_v2->next;

	res = DP_Error_N_sub( R_list_v1, R_list_v2, R_set_v->num, L, opt);

	//Release the related resources.
	R_set_release( R_set_v);

	return res;
}

/*
 *	The special handling procedure (retrieving the error) seg_after Merge-Span2.
 */
float error_retrieve( b_heap_t* b_h, int* tag)
{
	int i;
	float error_v, error_max;
	
	h_obj_t* h_obj_v;
	int* h_arr;
	
#ifndef WIN32
	struct rusage err_sta_t, err_end_t;
	get_cur_time( &err_sta_t);
#endif
	
	error_max = 0;
	h_obj_v = b_h->obj_arr;
	h_arr = b_h->h_arr;
	for( i=1; i<=b_h->rear; i++)
	{
		error_v = calc_error( h_obj_v[ h_arr[i]].R_node_v_sta, h_obj_v[ h_arr[i]].R_node_v_end);
		if( error_max < error_v)
			error_max = error_v;
	}
	
	if( b_h->rear != 0)
	{
		error_v = calc_error( h_obj_v[ h_arr[ b_h->rear]].R_node_v_sta1, h_obj_v[ h_arr[ b_h->rear]].R_node_v_end1);
		if( error_max < error_v)
			error_max = error_v;
	}
	
	//Additional checking.
	for( i=1; i<b_h->size; i++)
	{
		if( tag[ i])
		{
			error_v = calc_error( h_obj_v[ i].R_node_v_sta, h_obj_v[ i].R_node_v_end1);
			if( error_max < error_v)
				error_max = error_v;
		}
	}
	
#ifndef WIN32
	get_cur_time( &err_end_t);
	collect_time( &err_sta_t, &err_end_t, &stat_v.error_calc_time);
#endif

	return error_max;
}

/*
 *	General interface of the Min-Error problem.
 *
 *	@RN_graph_v indicates the road network.
 *	@L indicates the complexity requirement.
 *	@opt indicates the method to be used for this problem.
 *
 *	@return the minimized error.
 */
float Min_Error( RN_graph_t* RN_graph_v, int L, int opt)
{
	if( L >= RN_graph_v->edge_n)
		return 0;

	if( opt == 1)
	{
		printf( "Search-Error-N\n\n");
		return Search_Error_N( RN_graph_v, L, 1);
	}
	else if( opt == 2)
	{
		printf( "DP-Error-N\n\n");
		return DP_Error_N( RN_graph_v, L, 1);
	}
	else if( opt == 3)
	{
		printf( "Merge-Error\n\n");
		return Merge_Error( RN_graph_v, L);
	}
	else if( opt == 4)
	{
		int* tag;
		float error_v;
		b_heap_t* b_h;

		printf( "Merge-Span\n\n");
		
		//Merge-Span procedure.
		tag = NULL;
		b_h = Merge_Span2( RN_graph_v, L, tag);

		//Retrieve the error.
		if( b_h == NULL)
			error_v = FLT_MAX;
		else
			error_v = error_retrieve( b_h, tag);

		release_b_heap( b_h);
		if( tag != NULL)
			free( tag);

		return error_v;
	}
	else if( opt == 5)
	{
		printf( "Search-Error-N (basic version)\n\n");
		return Search_Error_N( RN_graph_v, L, 2);
	}
	else //opt == 6
	{
		printf( "DP-Error-N (basic version)\n\n");
		return DP_Error_N( RN_graph_v, L, 2);
	}	
}


//

/*
 *	Print a R_set_t structure.
 */
void print_R_set( R_set_t* R_set_v)
{
	R_list_t* R_list_v;
	R_node_t* R_node_v;

	printf( "# of roads:  %i\n\n", R_set_v->num);
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		R_node_v = R_list_v->head->next;
		
		while( R_node_v != NULL)
		{
			printf( "%i\t%i\t%f\n", R_node_v->id1, R_node_v->id2, R_node_v->slope);

			R_node_v = R_node_v->next;
		}

		printf( "\n\n");

		R_list_v = R_list_v->next;		
	}
}

/*
 *	Print a eps_g_t structure.
 */
void print_eps_g( eps_g_t* eps_g_v)
{
	int i;
	eps_g_node_t* g_node_v;

	for( i=0; i<eps_g_v->vertex_n; i++)
	{
		printf( "%i:\n", i+1);
		g_node_v = eps_g_v->head[ i].list->next;
		while( g_node_v != NULL)
		{
			printf( "\t%i\n", g_node_v->id + 1);

			g_node_v = g_node_v->next;
		}
		
		printf( "\n");
	}
}

/*
 *	Print a RN_graph_t structue.
 */
void print_RN_graph( RN_graph_t* RN_graph_v)
{
	int i;
	RN_graph_list_t* g_node_v;

	printf( "No. of vertices: %i\n", RN_graph_v->vertex_n);
	printf( "No. of edges:    %i\n", RN_graph_v->edge_n);

	for( i=0; i<RN_graph_v->vertex_n; i++)
	{
		printf( "%i  %f  %f\n", i, RN_graph_v->head_v[ i].loc_v.x, RN_graph_v->head_v[ i].loc_v.y);
		g_node_v = RN_graph_v->head_v[ i].list->next;
		while( g_node_v != NULL)
		{
			printf( "\t%i  %f  %f  %f\n", g_node_v->id, RN_graph_v->head_v[ g_node_v->id].loc_v.x,
														RN_graph_v->head_v[ g_node_v->id].loc_v.y,
														g_node_v->leng);

			g_node_v = g_node_v->next;
		}
		
		printf( "\n");
	}
}

/*
 *	Print the appr_table information.
 */
void print_appr_table( int** appr_table, int row_n, int col_n)
{
	int i, j;
	
	for( i=0; i<row_n; i++)
	{
		printf( "0  ");
		for( j=0; j<col_n; j++)
		{
			printf( "%i  ", appr_table[ i][ j]);
		}

		printf( "\n");
	}
	for( j=0; j<=col_n; j++)
		printf( "0  ");

	printf( "\n\n");
}

/*
 *	Print the search space information.
 */
void print_search_space( search_space_t* search_space_v)
{
	int i;
	
	for( i=0; i<search_space_v->size; i++)
		printf( "%f  ", search_space_v->s_space[ i]);
	printf( "\n\n");
}

/*
 *	Test the algorithms related to Min-Number.
 */
void test( )
{
	int num_res;
	float err_res;
	config_t* config_v;
	RN_graph_t* RN_graph_v;

	//Read the config.
	config_v = read_config( );

	//Read the road.
	RN_graph_v = read_data( config_v);

	//Perform the calculation.
	num_res = 0;
	err_res = 0;
	if( config_v->pro_opt == 1)
	{
		num_res = Min_Number( RN_graph_v, config_v->eps, config_v->alg_opt);		
		printf( "Min-Number:  %i\n", num_res);
	}
	else
	{
		err_res = Min_Error( RN_graph_v, config_v->L, config_v->alg_opt);
		printf( "Min-Error: %f\n", err_res);
	}

	//Release the related resources.
	free( config_v);
	RN_graph_release( RN_graph_v);
}


/*----------------------Modified on 26, Feb 2012-----------------------*/

/*
 *	Decompose an angle range [a, b] (a > b) into two: [a, \pi] and [0, b].
 */
slope_r_t* anlge_range_decompose( slope_r_t* slope_r_v)
{
	slope_r_t* slope_r_v1;

	slope_r_v1 = ( slope_r_t*)malloc( sizeof( slope_r_t));
	memset( slope_r_v1, 0, sizeof( slope_r_t));

	slope_r_v1->s_sta = slope_r_v->s_sta;
	slope_r_v1->s_end = 2 * PI;

	slope_r_v->s_sta = 0;

	return slope_r_v1;
}

/*
 *	modulo operator.
 */
float modulo_op( float t, float modulo)
{
	while( t < 0)
		t += modulo;
	while( t >= modulo)
		t -= modulo;

	return t;
}

/*
 *	Judge whether an value (angle) t falls in the range [a, b].
 *	
 *	@return 1 if yes; otherwise, return 0.
 */
int	is_in_slope_r( float t, float a, float b)
{
	if( modulo_op( t - a, 2 * PI) <= modulo_op( b - a, 2 * PI))
		return 1;
	return 0;
}

/*
 *	Calculate the intersection between two ranges: slope_r_v1 and slope_r_v2.
 *
 *	@slope_r_v3 (slope_r_v4) carries the intersection information if it is non-null.
 */
void slope_range_intersect( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2, slope_r_t* &slope_r_v3, slope_r_t* &slope_r_v4)
{
	float a1, b1, a2, b2;

	slope_r_v3 = NULL;
	slope_r_v4 = NULL;

	//Simple notations.
	a1 = slope_r_v1->s_sta;
	b1 = slope_r_v1->s_end;
	a2 = slope_r_v2->s_sta;
	b2 = slope_r_v2->s_end;

	//By cases.
	if( is_in_slope_r( a2, a1, b1))
	{
		//The intersection must not be empty.
		slope_r_v3 = ( slope_r_t*)malloc( sizeof( slope_r_t));
		memset( slope_r_v3, 0, sizeof( slope_r_t));


		/*s*/
		emp_stat_v.memory_v += sizeof( slope_r_t);
		if( emp_stat_v.memory_v > emp_stat_v.memory_max)
			emp_stat_v.memory_max = emp_stat_v.memory_v;
		/*s*/

		if( is_in_slope_r( b2, a2, b1))
		{
			//The intersection is [a2, b2] itself.			
			slope_r_v3->s_sta = a2;
			slope_r_v3->s_end = b2;
		}
		else if( is_in_slope_r( b2, a1, a2))
		{
			//The intersection is [a2, b1] and [a1, b2].
			slope_r_v4 = ( slope_r_t*)malloc( sizeof( slope_r_t));
			memset( slope_r_v4, 0, sizeof( slope_r_t));

			/*s*/
			emp_stat_v.memory_v += sizeof( slope_r_t);
			if( emp_stat_v.memory_v > emp_stat_v.memory_max)
				emp_stat_v.memory_max = emp_stat_v.memory_v;
			/*s*/

			slope_r_v3->s_sta = a2;
			slope_r_v3->s_end = b1;

			slope_r_v4->s_sta = a1;
			slope_r_v4->s_end = b2;
		}
		else	//b2 is in range [b1, a1]
		{
			//The intersection is [a2, b1].
			slope_r_v3->s_sta = a2;
			slope_r_v3->s_end = b1;
		}
	}
	else
	{
		if( is_in_slope_r( b2, a1, b1))
		{
			//The intersection is [a1, b2]
			slope_r_v3 = ( slope_r_t*)malloc( sizeof( slope_r_t));
			memset( slope_r_v3, 0, sizeof( slope_r_t));

			/*s*/
			emp_stat_v.memory_v += sizeof( slope_r_t);
			if( emp_stat_v.memory_v > emp_stat_v.memory_max)
				emp_stat_v.memory_max = emp_stat_v.memory_v;
			/*s*/

			slope_r_v3->s_sta = a1;
			slope_r_v3->s_end = b2;
		}
		else if( is_in_slope_r( b2, a2, a1))
		{
			//The intersection is empty.
		}
		else	//b2 is range (b1, a2)
		{
			//The intersection is [a1, b1].
			slope_r_v3 = ( slope_r_t*)malloc( sizeof( slope_r_t));
			memset( slope_r_v3, 0, sizeof( slope_r_t));

			/*s*/
			emp_stat_v.memory_v += sizeof( slope_r_t);
			if( emp_stat_v.memory_v > emp_stat_v.memory_max)
				emp_stat_v.memory_max = emp_stat_v.memory_v;
			/*s*/

			slope_r_v3->s_sta = a1;
			slope_r_v3->s_end = b1;
		}
	}
}

/*
 *	Allocate the memory for a slope_r_list_t struct.
 */
slope_r_list_t* slope_r_list_alloc( )
{
	slope_r_list_t* slope_r_list_v;

	slope_r_list_v = ( slope_r_list*)malloc( sizeof( slope_r_list_t));
	memset( slope_r_list_v, 0, sizeof( slope_r_list_t));

	slope_r_list_v->head = ( slope_r_node_t*)malloc( sizeof( slope_r_node_t));
	memset( slope_r_list_v->head, 0, sizeof( slope_r_node_t));

	slope_r_list_v->rear = slope_r_list_v->head;

	/*s*/
	emp_stat_v.memory_v += sizeof( slope_r_list_t) + sizeof( slope_r_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return slope_r_list_v;
}

/*
 *	Add a slope_r_t entry @slope_r_v to a slope_r_list_t struct @slope_r_list_v 
 *	the end entry is indicated by @rear.
 */	
void append_slope_r_list_entry( slope_r_list_t* slope_r_list_v, slope_r_t slope_r_v)
{
	slope_r_node_t* slope_r_node_v;

	slope_r_node_v = ( slope_r_node_t*)malloc( sizeof( slope_r_node_t));
	memset( slope_r_node_v, 0, sizeof( slope_r_node_t));
	
	slope_r_node_v->slope_r_v = slope_r_v;
	slope_r_list_v->rear->next = slope_r_node_v;
	slope_r_list_v->num ++;
	slope_r_list_v->rear = slope_r_node_v;


	/*s*/
	emp_stat_v.memory_v += sizeof( slope_r_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return;
}

/*
 *	Release a slope_r_list_t structure.
 */
void slope_r_list_release( slope_r_list_t* slope_r_list_v)
{
	slope_r_node_t* slope_r_node_v, *slope_r_node_v1;

	if( slope_r_list_v == NULL)
		return;
	
	slope_r_node_v = slope_r_list_v->head;
	while( slope_r_node_v != NULL)
	{
		slope_r_node_v1 = slope_r_node_v->next;

		free( slope_r_node_v);

		slope_r_node_v = slope_r_node_v1;
	}

	/*s*/
	emp_stat_v.memory_v -= sizeof( slope_r_node_t) * ( slope_r_list_v->num + 1)
							+ sizeof( slope_r_list_t);
	/*s*/

	free( slope_r_list_v);
}

/*
 *	Intersect the slope ranges maintained in slope_r_list and a slope range.
 *
 *	 The current implementation could be shorten to a certain extent.
 *
 *	@return NULL if the intersection is empty; otherwise return the intersection.
 */
slope_r_list_t* range_intersect( slope_r_list_t* slope_r_list_v, slope_r_t* slope_r_v)
{
	slope_r_t* slope_r_v1, *slope_r_v2;
	slope_r_list_t* slope_r_list_v1;
	slope_r_node_t* slope_r_node_v;

	/*t/
	printf( "Before intersection:\n");
	print_slope_r_list( slope_r_list_v);
	printf( "[%f, %f]\n", slope_r_v->s_sta, slope_r_v->s_end);
	/*t*/

	slope_r_list_v1 = slope_r_list_alloc( );

	slope_r_node_v = slope_r_list_v->head->next;
	while( slope_r_node_v != NULL)
	{
		slope_range_intersect( &slope_r_node_v->slope_r_v, slope_r_v, slope_r_v1, slope_r_v2);
		if( slope_r_v1 != NULL)
		{
			append_slope_r_list_entry( slope_r_list_v1, *slope_r_v1);

			free( slope_r_v1);
			
			/*s*/
			emp_stat_v.memory_v -= sizeof( slope_r_t);
			/*s*/
		}
		if( slope_r_v2 != NULL)
		{
			append_slope_r_list_entry( slope_r_list_v1, *slope_r_v2);

			free( slope_r_v2);
			
			/*s*/
			emp_stat_v.memory_v -= sizeof( slope_r_t);
			/*s*/
		}

		slope_r_node_v = slope_r_node_v->next;
	}

	/*t/
	printf( "After intersection:\n");
	print_slope_r_list( slope_r_list_v1);
	printf( "\n");

	if( slope_r_list_v1 != NULL && slope_r_list_v1->num != 0)
	{
		if( slope_r_list_v1->head->next->slope_r_v.s_sta > 2.83 &&
			slope_r_list_v1->head->next->slope_r_v.s_sta < 2.84)
			printf( "");
	}
	/*t*/

	/*s*/
	emp_stat_v.cap_n ++;	

	if( slope_r_list_v->num > emp_stat_v.C.max)
		emp_stat_v.C.max = slope_r_list_v->num;
	if( slope_r_list_v->num < emp_stat_v.C.min)
		emp_stat_v.C.min = slope_r_list_v->num;

	emp_stat_v.C.sum += slope_r_list_v->num;
	/*s*/

	if( slope_r_list_v1->num == 0)
	{
		slope_r_list_release( slope_r_list_v1);
		slope_r_list_v1 = NULL;
	}
	
	return slope_r_list_v1;
}

/*
 *	Check whether a slope is within a range list.
 */
int is_in_slope_r_list( slope_r_list_t* slope_r_list_v, float slope)
{
	slope_r_node_t* slope_r_node_v;
	slope_r_t slope_r_v;

	if( slope_r_list_v == NULL)
		return 0;

	slope_r_node_v = slope_r_list_v->head->next;
	while( slope_r_node_v != NULL)
	{
		slope_r_v = slope_r_node_v->slope_r_v;
		
		if( is_in_slope_r( slope, slope_r_v.s_sta, slope_r_v.s_end))
			return 1;

		slope_r_node_v = slope_r_node_v->next;
	}

	return 0;
}

/*
 *	Calculate the error table of \epsilon( \overline{ v_iv_j}) for 0 <= i < j <= n.
 *	
 *	The space cost of this implementation is still O(n^2).
 */
int** calc_appr_table2( R_list_t* R_list_v, float eps)
{
	int i, j, seg_n;
	int** error_tab;
	float slope_v, slope_v1;

	slope_r_list_t* slope_r_list_v1, *slope_r_list_v2;
	slope_r_t slope_r_v;

	R_node_t* R_node_v1, *R_node_v2;

	seg_n = R_list_v->size;

	error_tab = ( int**)malloc( seg_n * sizeof( int*));
	memset( error_tab, 0, seg_n * sizeof( int*));

	R_node_v1 = R_list_v->head->next;
	//for( i=0; i<seg_n; i++)
	i = 0;
	while( R_node_v1 != NULL)
	{
		error_tab[ i] = ( int*)malloc( seg_n * sizeof( int));
		memset( error_tab[ i], 0, seg_n * sizeof( int));

		//The fdr of the first segment.
		slope_v = R_node_v1->slope;
		slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
		slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

		//Maintain the slope range list.
		slope_r_list_v1 = slope_r_list_alloc( );
		append_slope_r_list_entry( slope_r_list_v1, slope_r_v);

		error_tab[ i][ i] = 1;

		R_node_v2 = R_node_v1->next;
		//for( j=i+1; j<seg_n; j++)
		j = i + 1;
		while( R_node_v2 != NULL)
		{
			slope_v = R_node_v2->slope;
			slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
			slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

			slope_r_list_v2 = range_intersect( slope_r_list_v1, &slope_r_v);

			slope_r_list_release( slope_r_list_v1);
			slope_r_list_v1 = NULL;

			if( slope_r_list_v2 == NULL)
				break;
			else	//slope_r_list_v2 != NULL.
			{
				//bug.
				slope_v1 = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);
				if( is_in_slope_r_list( slope_r_list_v2, slope_v1))
					error_tab[ i][ j] = 1;

				slope_r_list_v1 = slope_r_list_v2;
			}

			R_node_v2 = R_node_v2->next;
			j++;
		}//

		slope_r_list_release( slope_r_list_v1);
		R_node_v1 = R_node_v1->next;
		i++;
	}//

	/*s*/
	emp_stat_v.memory_v += sizeof( int*) * seg_n + sizeof( int) * seg_n * seg_n;
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return error_tab;
}

/*
 *	Search the closest key to the provided key in the bst.
 *
 *	@T is the bst.
 *	@key is provided key.
 *	
 *	@return the closest gap from the key.
 */
KEY_TYPE closest_gap( bst_t* T, KEY_TYPE key)
{
	float gap1, gap2;
	bst_node_t* pre, *suc, *x, *y, *z;

	//Search for the appropriate location for the key.
	y = NULL;
	x = T->root;
	while( x != NULL)
	{
		y = x;

		if( key < x->key)
			x = x->left;
		else
			x = x->right;
	}

	//Search for the successor.
	z = x;
	suc = y;
	while( suc != NULL && z == suc->right)
	{
		z = suc;
		suc = suc->p;
	}	

	//Search for the predecessor.
	z = x;
	pre = y;
	while( pre != NULL && z == pre->left)
	{
		z = pre;
		pre = pre->p;
	}

	//Decide the closest key.
	/*t*/
	if( pre == NULL && suc == NULL)
	{
		fprintf( stderr, "Both predecessor and successor are empty.\nSomething wrong!\n");
		exit( 0);
	}
	/*t*/

	if( pre == NULL)
		gap1 = modulo_op( key - T->max, 2 * PI);
	else
		gap1 = key - pre->key;

	if( suc == NULL)
		gap2 = modulo_op( T->min - key, 2 * PI);
	else
		gap2 = suc->key - key;
	
	return gap1 <= gap2 ? gap1 : gap2;
}

/*
 *	Calculate the error of a slope based on the boundary information.
 */
float check_error_boundary( float b_1, float b_2, float slope)
{
	float error_v1, error_v2;
	
	error_v1 = modulo_op( slope - b_1, 2 * PI);
	error_v2 = modulo_op( b_2 - slope, 2 * PI);

	return error_v1 >= error_v2 ? error_v1 : error_v2;
}

/* 
 *	Allocate memory for a span_m_t structure.
 
span_m_t* span_m_alloc( )
{
	span_m_t* span_m_v;

	span_m_v = ( span_m_t*)malloc( sizeof( span_m_t));
	span_m_v->min_1 = span_m_v->min_2 = FLT_MAX;
	span_m_v->max_1 = span_m_v->max_2 = -FLT_MAX;

	return span_m_v;
}
*/

/*
 *	Initialize a span_m_t strucute.
 */
void span_m_ini( span_m_t* span_m_v)
{
	span_m_v->min_1 = span_m_v->min_2 = FLT_MAX;
	span_m_v->max_1 = span_m_v->max_2 = -FLT_MAX;
}

/*
 *	Update the information of a span_m_t structure.
 */
void span_m_update( span_m_t* span_m_v, float slope)
{
	if( slope >= 0 && slope < PI)
	{
		if( span_m_v->min_1 > slope)
			span_m_v->min_1 = slope;
		if( span_m_v->max_1 < slope)
			span_m_v->max_1 = slope;
	}
	else
	{
		if( span_m_v->min_2 > slope)
			span_m_v->min_2 = slope;
		if( span_m_v->max_2 < slope)
			span_m_v->max_2 = slope;
	}
}

/*
 *	Calculate the span information based on the span_m_t data.
 */
slope_r_t* calc_span( span_m_t* span_m_v)
{
	float min_3, max_3;
	slope_r_t* slope_r_v;
	
	slope_r_v = ( slope_r_t*)malloc( sizeof( slope_r_t));
	memset( slope_r_v, 0, sizeof( slope_r_t));

	//Judge the span.
	if( span_m_v->min_1 == FLT_MAX)
	{
		//The span is [min_2, max_2].
		slope_r_v->s_sta = span_m_v->min_2;
		slope_r_v->s_end = span_m_v->max_2;
	}
	else if( span_m_v->min_2 == FLT_MAX)
	{
		//The span is [min_1, max_1].
		slope_r_v->s_sta = span_m_v->min_1;
		slope_r_v->s_end = span_m_v->max_1;
	}
	else	
	{
		min_3 = span_m_v->min_1 + PI;			
		max_3 = span_m_v->max_1 + PI;
		
		if( min_3 >= span_m_v->min_2 && min_3 <= span_m_v->max_2 ||
			max_3 >= span_m_v->min_2 && max_3 <= span_m_v->max_2)
		{
			//[min_3, max_3] and [min_2, max_2] are non-disjoint.
			//The span is the whole range [0, 2\pi).
			slope_r_v->s_sta = 0;
			slope_r_v->s_end = 2 * PI;
		}
		else	
		{
			//[min_3, max_3] and [min_2, max_2] are disjoint.
			if( max_3 < span_m_v->min_2)
			{
				//The span is [min_2, max_1].
				slope_r_v->s_sta = span_m_v->min_2;
				slope_r_v->s_end = span_m_v->max_1;
			}
			else	//min_3 > max_2
			{
				//The span is [min_1, max_2].
				slope_r_v->s_sta = span_m_v->min_1;
				slope_r_v->s_end = span_m_v->max_2;
			}
		}
	}

	return slope_r_v;
}

/*
 *	Calculate the error based on the span_m_t data.
 */
float check_error_span( bst_t* T, span_m_t* span_m_v, float slope)
{
	float slope_1, error_v;
	slope_r_t* slope_r_v;

	//Calculate the span.
	slope_r_v = calc_span( span_m_v);

	//Calculate the error based on the span information.
	if( slope_r_v->s_sta == 0 && slope_r_v->s_end == 2 * PI)
	{
		//Case 1: The extent of the span is 2* PI.
		slope_1 = modulo_op( slope + PI, 2 * PI);
		error_v = PI - closest_gap( T, slope_1);
	}
	else	
	{
		//Case 2: The extent of the span is smaller than 2* PI.
		error_v = check_error_boundary( slope_r_v->s_sta, slope_r_v->s_end, slope);
	}
	
	free( slope_r_v);

	return error_v;
}

/*
 *	Construct the search space for Search-Error-Road algorithm [O(n^2*log n) implementation].
 * 
 *	The method is to incrementally maintain the "span-information".
 *
float* search_space_R2( R_list_t* R_list_v)
{
	int seg_n, s_size, cnt;
	float slope;
	float* s_space;
	
	R_node_t* R_node_v1, *R_node_v2;
	bst_t* T;
	bst_node_t* bst_node_v;
	span_m_t* span_m_v;

	seg_n = R_list_v->size;
	if( seg_n == 0)
	{
		printf( "The road contains nothing when constructing the search space.\n");
		return NULL;
	}

	s_size = seg_n * ( seg_n + 1) / 2;
	s_space = ( float*)malloc( sizeof( float) * s_size);
	memset( s_space, 0, sizeof( float) * s_size);

	cnt = 0;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1 != NULL)
	{
		T = bst_ini( );
		span_m_v = ( span_m_t*)malloc( sizeof( span_m_t));
		span_m_ini( span_m_v);

		R_node_v2 = R_node_v1;
		while( R_node_v2 != NULL)
		{
			//Maintain the span information.
			bst_node_v = ( bst_node_t*)malloc( sizeof( bst_node_t));
			memset( bst_node_v, 0, sizeof( bst_node_t));
			bst_node_v->key = R_node_v2->slope;
			bst_insert( T, bst_node_v);

			span_m_update( span_m_v, R_node_v2->slope);

			//Calculate the slope.
			slope = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);

			//Calculate the error.
			s_space[ cnt++] = check_error_span( T, span_m_v, slope);

			//t
			if( R_node_v2 == R_node_v1 && s_space[ cnt-1] != 0)
			{
				fprintf( stderr, "The error of approximating itself is not 0 [calc_abs_slope].\n");
				exit( 0);
			}
			//t

			R_node_v2 = R_node_v2->next;
		}

		bst_release( T);
		free( span_m_v);

		R_node_v1 = R_node_v1->next;
	}

	//t
	if( cnt != s_size)
	{
		fprintf( stderr, "Inconsistency of the size information of the search space.\n");
		exit( 0);
	}
	//t

	return s_space;
}
*/

/*
 *	Construct the search space for Search-Error-N algorithm (based on search_space_R2).
 *
search_space_t* search_space_N2( RN_graph_t* RN_graph_v)
{
	int i, s_size, cnt, seg_n, s_size_tmp;
	float* s_space, *s_space_tmp;
	search_space_t* search_space_v;
	R_set_t* R_set_v;
	R_list_t* R_list_v;

	search_space_v = ( search_space_t*)malloc( sizeof( search_space_t));
	memset( search_space_v, 0, sizeof( search_space_t));

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);
	
	//Collect the size information of the search space.
	cnt = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		cnt += seg_n * ( seg_n + 1) / 2;

		R_list_v = R_list_v->next;
	}

	//Allocate the memory.
	s_size = cnt;
	s_space = ( float*)malloc( sizeof( float) * s_size);
	memset( s_space, 0, sizeof( float) * s_size);

	//Collect the possible error information.
	cnt = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		s_size_tmp = seg_n * ( seg_n + 1) / 2;
		s_space_tmp = search_space_R2( R_list_v);
		for( i=0; i<s_size_tmp; i++)
			s_space[ cnt++] = s_space_tmp[ i];

		R_list_v = R_list_v->next;

		free( s_space_tmp);
	}

	search_space_v->s_space = s_space;
	search_space_v->size = s_size;

	//Release the related resources.
	R_set_release( R_set_v);

	return search_space_v;
}
*/

/*
 *	Construct the search space for Search-Error-Road algorithm [O(n^2*log n) implementation].
 *
 *	What's the difference from search_space_R2?
 *
 *
 */
float* search_space_R3( R_list_t* R_list_v)
{
	int seg_n, s_size, cnt;
	float slope;
	float* s_space;
	
	R_node_t* R_node_v1, *R_node_v2;
	bst_t* T;
	bst_node_t* bst_node_v;
	span_m_t* span_m_v;

	seg_n = R_list_v->size;
	if( seg_n == 0)
	{
		printf( "The road contains nothing when constructing the search space.\n");
		return NULL;
	}

	s_size = seg_n * ( seg_n - 1) / 2 + 1;
	s_space = ( float*)malloc( sizeof( float) * s_size);
	memset( s_space, 0, sizeof( float) * s_size);

	/*s/
	stat_v.memory_v += sizeof( float) * s_size);
	if( stat_v.memory_max < stat_v.memory_v)
		stat_v.memory_max = stat_v.memory_v;
	/*s*/

	//For all the line segments approximated by themselves.
	s_space[ 0] = 0;

	cnt = 1;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1 != NULL)
	{
		T = bst_ini( );
		span_m_v = ( span_m_t*)malloc( sizeof( span_m_t));
		span_m_ini( span_m_v);

		bst_node_v = ( bst_node_t*)malloc( sizeof( bst_node_t));
		memset( bst_node_v, 0, sizeof( bst_node_t));
		bst_node_v->key = R_node_v1->slope;
		bst_insert( T, bst_node_v);

		span_m_update( span_m_v, R_node_v1->slope);

		R_node_v2 = R_node_v1->next;
		while( R_node_v2 != NULL)
		{
			//Maintain the span information.
			bst_node_v = ( bst_node_t*)malloc( sizeof( bst_node_t));
			memset( bst_node_v, 0, sizeof( bst_node_t));
			bst_node_v->key = R_node_v2->slope;
			bst_insert( T, bst_node_v);

			span_m_update( span_m_v, R_node_v2->slope);

			//Calculate the slope.
			slope = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);

			//Calculate the error.
			s_space[ cnt++] = check_error_span( T, span_m_v, slope);

			/*t*/
			if( R_node_v2 == R_node_v1 && s_space[ cnt-1] != 0)
			{
				fprintf( stderr, "The error of approximating itself is not 0 [calc_abs_slope].\n");
				exit( 0);
			}
			/*t*/

			R_node_v2 = R_node_v2->next;
		}

		/*s*/
		stat_v.memory_v += T->m_size;
		if( stat_v.memory_max < stat_v.memory_v)
			stat_v.memory_max = stat_v.memory_v;
		stat_v.memory_v -= T->m_size;
		/*s*/

		bst_release( T);
		free( span_m_v);

		R_node_v1 = R_node_v1->next;
	}

	/*t*/
	if( cnt != s_size)
	{
		fprintf( stderr, "Inconsistency of the size information of the search space.\n");
		exit( 0);
	}
	/*t*/

	return s_space;
}

/*
 *	Construct the search space for Search-Error-N algorithm (based on search_space_R2).
 */
search_space_t* search_space_N3( R_set_t* R_set_v)
{
	int i, s_size, cnt, seg_n, s_size_tmp;
	float* s_space, *s_space_tmp;
	search_space_t* search_space_v;
	R_list_t* R_list_v;

	search_space_v = ( search_space_t*)malloc( sizeof( search_space_t));
	memset( search_space_v, 0, sizeof( search_space_t));

	/*t/
	print_R_set( R_set_v);
	/*t*/
	
	//Collect the size information of the search space.
	cnt = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		cnt += seg_n * ( seg_n - 1) / 2 + 1;

		R_list_v = R_list_v->next;
	}

	//Allocate the memory.
	s_size = cnt;
	s_space = ( float*)malloc( sizeof( float) * s_size);
	memset( s_space, 0, sizeof( float) * s_size);

	//Collect the possible error information.
	cnt = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		s_size_tmp = seg_n * ( seg_n - 1) / 2 + 1;
		s_space_tmp = search_space_R3( R_list_v);
		for( i=0; i<s_size_tmp; i++)
			s_space[ cnt++] = s_space_tmp[ i];

		R_list_v = R_list_v->next;

		free( s_space_tmp);
	}

	search_space_v->s_space = s_space;
	search_space_v->size = s_size;

	/*s*/
	stat_v.memory_v += sizeof( search_space_t) + sizeof( float) * s_size;
	if( stat_v.memory_max < stat_v.memory_v)
		stat_v.memory_max = stat_v.memory_v;
	/*s*/

	return search_space_v;
}


/*
 *	The implementation of Merge-Error.
 */
float Merge_Error( RN_graph_t* RN_graph_v, int L)
{
	int i, top_n, pre_t, next_t, pair_n, cnt, edge_n;
	float max_t;

	R_set_t* R_set_v;
	R_list_t* R_list_v;
	R_node_t* R_node_v1, *R_node_v2;

	b_heap_t* b_h;
	h_obj_t* obj_v, *obj_v1;

	edge_n = RN_graph_v->edge_n;
	/*
	if( L >= edge_n)
		return 0;
	*/

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);
	if( L < R_set_v->num)
	{
		R_set_release( R_set_v);
		return FLT_MAX;
	}

	//Collect the number of pairs.
	pair_n = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		pair_n += R_list_v->size - 1;
		R_list_v = R_list_v->next;
	}

	//Initialize the heap (error information).
	b_h = alloc_b_heap( pair_n + 1);

	cnt = 1;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		R_node_v1 = R_list_v->head->next;
		R_node_v2 = R_node_v1->next;
		while( R_node_v2 != NULL)
		{
			//The span between R_node_v1 and R_node_v2.
			obj_v = &b_h->obj_arr[ cnt];
			obj_v->id = cnt;

			obj_v->R_node_v_sta = R_node_v1;
			obj_v->R_node_v_end = R_node_v2;
//			obj_v->R_node_v_sta2 = R_node_v2;
//			obj_v->R_node_v_end2 = R_node_v2;

			obj_v->key = calc_error( obj_v->R_node_v_sta, obj_v->R_node_v_end);

			if( R_node_v1 ==  R_list_v->head->next)
				obj_v->pre = 0;				//No predecessor spans.
			else
				obj_v->pre = cnt - 1;
			if( R_node_v2->next == NULL)
				obj_v->next = 0;			//No successor spans.
			else
				obj_v->next = cnt + 1;

			//Insert the span into the heap.
			insert_b_heap( b_h, cnt);

			cnt ++;
			R_node_v1 = R_node_v1->next;
			R_node_v2 = R_node_v1->next;
		}

		R_list_v = R_list_v->next;
	}
	
	//Perform the 'merge' operation (size - L) times.
	max_t = 0;
	for( i=0; i<edge_n-L; i++)
	{
		top_n = get_top( b_h);
	
		obj_v = &b_h->obj_arr[ top_n];
		if( max_t < obj_v->key)
			max_t = obj_v->key;

		//Merge.
		pre_t = obj_v->pre;
		if( pre_t > 0)
		{
			//Update the previous span.
			obj_v1 = &b_h->obj_arr[ pre_t];
			obj_v1->R_node_v_end = obj_v->R_node_v_end;
			obj_v1->next = obj_v->next;
			obj_v1->key = calc_error( obj_v1->R_node_v_sta, obj_v1->R_node_v_end);

			update_key( b_h, obj_v1->id);
		}

		next_t = obj_v->next;
		if( next_t > 0)
		{
			//Update the next potential span.
			obj_v1 = &b_h->obj_arr[ next_t];
			obj_v1->R_node_v_sta = obj_v->R_node_v_sta;
			obj_v1->pre = obj_v->pre;
			obj_v1->key = calc_error( obj_v1->R_node_v_sta, obj_v1->R_node_v_end);

			update_key( b_h, obj_v1->id);
		}
	}

	//Release the data structures.
	release_b_heap( b_h);
	R_set_release( R_set_v);

	return max_t;
}

/*
 *	Combine two spans into a single one.
 */
span_m_t* span_combine( span_m_t* span_m_v1, span_m_t* span_m_v2)
{
	span_m_t* span_m_v;

	span_m_v = ( span_m_t*)malloc( sizeof( span_m_t));
	span_m_ini( span_m_v);
	
	span_m_v->min_1 = min( span_m_v1->min_1, span_m_v2->min_1);
	span_m_v->max_1 = max( span_m_v1->max_1, span_m_v2->max_1);

	span_m_v->min_2 = min( span_m_v1->min_2, span_m_v2->min_2);
	span_m_v->max_2 = max( span_m_v1->max_2, span_m_v2->max_2);

	return span_m_v;
}

/*
 *	The implementation of Merge-Span.
 */
float Merge_Span( RN_graph_t* RN_graph_v, int L)
{
	int i, top_n, pre_t, next_t, pair_n, cnt, edge_n;
	float max_t;

	R_set_t* R_set_v;
	R_list_t* R_list_v;
	R_node_t* R_node_v1, *R_node_v2;

	b_heap_t* b_h;
	h_obj_t* obj_v, *obj_v1;
	
	span_m_t* span_m_v;
	slope_r_t* slope_r_v;

	edge_n = RN_graph_v->edge_n;
	/*
	if( L >= edge_n)
		return 0;
	*/

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);
	if( L < R_set_v->num)
	{
		R_set_release( R_set_v);
		return FLT_MAX;
	}

	//Collect the number of pairs.
	pair_n = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		pair_n += R_list_v->size - 1;
		R_list_v = R_list_v->next;
	}

	//Initialize the heap (error information).
	b_h = alloc_b_heap( pair_n + 1);

	cnt = 1;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		R_node_v1 = R_list_v->head->next;
		R_node_v2 = R_node_v1->next;
		while( R_node_v2 != NULL)
		{
			//The span between R_node_v1 and R_node_v2.
			obj_v = &b_h->obj_arr[ cnt];
			obj_v->id = cnt;

			span_m_ini( &obj_v->span_m_v);
			span_m_ini( &obj_v->span_m_v);
			span_m_update(  &obj_v->span_m_v, R_node_v1->slope);
			span_m_update(  &obj_v->span_m_v, R_node_v2->slope);

			slope_r_v = calc_span( &obj_v->span_m_v);
			obj_v->key = modulo_op( slope_r_v->s_end - slope_r_v->s_sta, 2 * PI);

			if( R_node_v1 ==  R_list_v->head->next)
				obj_v->pre = 0;				//No predecessor spans.
			else
				obj_v->pre = cnt - 1;
			if( R_node_v2->next == NULL)
				obj_v->next = 0;			//No successor spans.
			else
				obj_v->next = cnt + 1;

			//Insert the span into the heap.
			insert_b_heap( b_h, cnt);

			cnt ++;
			R_node_v1 = R_node_v1->next;
			R_node_v2 = R_node_v1->next;

			free( slope_r_v);
		}

		R_list_v = R_list_v->next;
	}
	
	//Perform the 'merge' operation (size - L) times.
	max_t = 0;
	for( i=0; i<edge_n-L; i++)
	{
		top_n = get_top( b_h);
	
		obj_v = &b_h->obj_arr[ top_n];
		if( max_t < obj_v->key)
			max_t = obj_v->key;

		//Merge.
		pre_t = obj_v->pre;
		if( pre_t > 0)
		{
			//Update the previous span.
			obj_v1 = &b_h->obj_arr[ pre_t];
			obj_v1->next = obj_v->next;

			span_m_v = span_combine( &obj_v1->span_m_v, &obj_v->span_m_v);
			obj_v1->span_m_v.min_1 = span_m_v->min_1;
			obj_v1->span_m_v.max_1 = span_m_v->max_1;
			obj_v1->span_m_v.min_2 = span_m_v->min_2;
			obj_v1->span_m_v.max_2 = span_m_v->max_2;
			
			slope_r_v = calc_span( &obj_v1->span_m_v);
			obj_v1->key = modulo_op( slope_r_v->s_end - slope_r_v->s_sta, 2 * PI);
			update_key( b_h, obj_v1->id);

			free( span_m_v);
			free( slope_r_v);
		}

		next_t = obj_v->next;
		if( next_t > 0)
		{
			//Update the next potential span.
			obj_v1 = &b_h->obj_arr[ next_t];
			obj_v1->pre = obj_v->pre;

			span_m_v = span_combine( &obj_v1->span_m_v, &obj_v->span_m_v);
			obj_v1->span_m_v.min_1 = span_m_v->min_1;
			obj_v1->span_m_v.max_1 = span_m_v->max_1;
			obj_v1->span_m_v.min_2 = span_m_v->min_2;
			obj_v1->span_m_v.max_2 = span_m_v->max_2;
			
			slope_r_v = calc_span( &obj_v1->span_m_v);
			obj_v1->key = modulo_op( slope_r_v->s_end - slope_r_v->s_sta, 2 * PI);
			update_key( b_h, obj_v1->id);

			free( span_m_v);
			free( slope_r_v);
		}
	}

	//Release the data structures.
	release_b_heap( b_h);
	R_set_release( R_set_v);

	return max_t;
}

/*
 *	Enhanced version of Merge_Span.
 *		-return the approximation error information, instead of only the span information.
 */
b_heap_t* Merge_Span2( RN_graph_t* RN_graph_v, int L, int* &tag)
{
	int i, top_n, pre_t, next_t, pair_n, cnt, edge_n;

	R_set_t* R_set_v;
	R_list_t* R_list_v;
	R_node_t* R_node_v1, *R_node_v2;

	b_heap_t* b_h;
	h_obj_t* obj_v, *obj_v1;
	
	span_m_t* span_m_v;
	slope_r_t* slope_r_v;

	edge_n = RN_graph_v->edge_n;
	/*
	if( L >= edge_n)
		return 0;
	*/

	//Decompose the road network.
	R_set_v = RN_decompose( RN_graph_v);
	if( L < R_set_v->num)
	{
		R_set_release( R_set_v);
		return NULL;
	}

	//Collect the number of pairs.
	pair_n = 0;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		pair_n += R_list_v->size - 1;
		R_list_v = R_list_v->next;
	}

	//Initialize the heap (error information).
	b_h = alloc_b_heap( pair_n + 1);

	tag = ( int*)malloc( sizeof( int) * (pair_n + 1));
	memset( tag, 0, sizeof( int) * ( pair_n + 1));

	cnt = 1;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		R_node_v1 = R_list_v->head->next;
		R_node_v2 = R_node_v1->next;
		while( R_node_v2 != NULL)
		{
			//The span between R_node_v1 and R_node_v2.
			obj_v = &b_h->obj_arr[ cnt];
			obj_v->id = cnt;

			span_m_ini( &obj_v->span_m_v);
			span_m_ini( &obj_v->span_m_v);
			span_m_update(  &obj_v->span_m_v, R_node_v1->slope);
			span_m_update(  &obj_v->span_m_v, R_node_v2->slope);

			slope_r_v = calc_span( &obj_v->span_m_v);
			obj_v->key = modulo_op( slope_r_v->s_end - slope_r_v->s_sta, 2 * PI);

			if( R_node_v1 ==  R_list_v->head->next)
				obj_v->pre = 0;				//No predecessor spans.
			else
				obj_v->pre = cnt - 1;
			if( R_node_v2->next == NULL)
				obj_v->next = 0;			//No successor spans.
			else
				obj_v->next = cnt + 1;

			//For Merge_Span2.
			obj_v->R_node_v_sta = R_node_v1;
			obj_v->R_node_v_end = R_node_v1;
			obj_v->R_node_v_sta1 = R_node_v2;
			obj_v->R_node_v_end1 = R_node_v2;

			//Insert the span into the heap.
			insert_b_heap( b_h, cnt);

			cnt ++;
			R_node_v1 = R_node_v1->next;
			R_node_v2 = R_node_v1->next;

			free( slope_r_v);
		}

		R_list_v = R_list_v->next;
	}
	
	//Perform the 'merge' operation (size - L) times.
	for( i=0; i<edge_n-L; i++)
	{
		top_n = get_top( b_h);
		obj_v = &b_h->obj_arr[ top_n];		

		//Merge.
		pre_t = obj_v->pre;
		if( pre_t > 0)
		{
			//Update the previous span.
			obj_v1 = &b_h->obj_arr[ pre_t];
			obj_v1->next = obj_v->next;

			span_m_v = span_combine( &obj_v1->span_m_v, &obj_v->span_m_v);
			obj_v1->span_m_v.min_1 = span_m_v->min_1;
			obj_v1->span_m_v.max_1 = span_m_v->max_1;
			obj_v1->span_m_v.min_2 = span_m_v->min_2;
			obj_v1->span_m_v.max_2 = span_m_v->max_2;
			
			slope_r_v = calc_span( &obj_v1->span_m_v);
			obj_v1->key = modulo_op( slope_r_v->s_end - slope_r_v->s_sta, 2 * PI);

			//For Merge_Span2.
			obj_v1->R_node_v_end1 = obj_v->R_node_v_end1;

			update_key( b_h, obj_v1->id);

			free( span_m_v);
			free( slope_r_v);
		}

		next_t = obj_v->next;
		if( next_t > 0)
		{
			//Update the next potential span.
			obj_v1 = &b_h->obj_arr[ next_t];
			obj_v1->pre = obj_v->pre;

			span_m_v = span_combine( &obj_v1->span_m_v, &obj_v->span_m_v);
			obj_v1->span_m_v.min_1 = span_m_v->min_1;
			obj_v1->span_m_v.max_1 = span_m_v->max_1;
			obj_v1->span_m_v.min_2 = span_m_v->min_2;
			obj_v1->span_m_v.max_2 = span_m_v->max_2;
			
			slope_r_v = calc_span( &obj_v1->span_m_v);
			obj_v1->key = modulo_op( slope_r_v->s_end - slope_r_v->s_sta, 2 * PI);

			//For Merge_Span2.
			obj_v1->R_node_v_sta = obj_v->R_node_v_sta;

			update_key( b_h, obj_v1->id);

			free( span_m_v);
			free( slope_r_v);
		}

		//bug.
		if( pre_t <= 0 && next_t <= 0)
		{
			//Tag the error information.
			tag[ obj_v->id] = 1;
		}
	}

	//Release the resources.
	R_set_release( R_set_v);

	return b_h;
}

/*
 *	Retrievet the shortest path information [not just the length of the path like calc_SP].
 *
 *	@eps_g_v, s_id and d_id are the same as for calc_SP1.
 *	@RN_graph_v_sim, 
 */
int calc_SP_RN( RN_graph_t* RN_graph_v_sim, RN_graph_t* RN_graph_v_ori, eps_g_t* eps_g_v, int s_id, int d_id)
{
	int i, vertex_n, sta, rear, c_id, n_id, rtn, pre_t, cur_t, id1, id2;
	int* que, *level, *pre;

	eps_g_node_t* g_node_v;
	RN_graph_list_t* RN_graph_list_v;

	vertex_n = eps_g_v->vertex_n;
	que = ( int*)malloc( sizeof( int) * vertex_n);
	level = ( int*)malloc( sizeof( int) * vertex_n);
	pre = ( int*)malloc( sizeof( int) * vertex_n);
	memset( que, 0, sizeof( int) * vertex_n);
	memset( level, 0, sizeof( int) * vertex_n);
	memset( pre, 0, sizeof( int) * vertex_n);

	/*s*/
	stat_v.memory_v += sizeof( int) * vertex_n * 3;
	if( stat_v.memory_max < stat_v.memory_v)
		stat_v.memory_max = stat_v.memory_v;
	/*s*/
	
	for( i=0; i<vertex_n; i++)
		pre[ i] = -2;

	rtn = 0;

	sta = rear = -1;
	que[ ++rear] = s_id;
	pre[ s_id] = -1; 
	while( sta < rear)
	{
		c_id = que[ ++sta];

		g_node_v = eps_g_v->head[ c_id].list->next;
		while( g_node_v != NULL)
		{
			n_id = g_node_v->id;
			if( n_id == d_id)
			{
				//The shortest path found.
				rtn = level[ c_id] + 1;
				pre[ n_id] = c_id;

				break;
			}

			if( pre[ n_id] == -2)
			{
				/*t*/
				if( rear == vertex_n - 1)
				{
					//The destination must be reachable.
					fprintf( stderr, "Incorrect eps_g_v [calc_SP_RN].\n");
					exit( 0);
				}
				/*t*/

				que[ ++rear] = n_id;
				level[ n_id] = level[ c_id] + 1;
				pre[ n_id] = c_id;
			}
			
			g_node_v = g_node_v->next;
		}//while( g_node_v)

		if( rtn != 0)
		{
			//The shortest path has been found.
			//Trace the shortest path.
			cur_t = d_id;
			pre_t = pre[ cur_t];
			while( pre_t != -1)
			{				
				id1 = eps_g_v->head[ cur_t].ori_id;
				id2 = eps_g_v->head[ pre_t].ori_id;

				//Construct the edge in the simplified road network.
				//1. id1->id2.
				RN_graph_list_v = ( RN_graph_list_t*)malloc( sizeof( RN_graph_list_t));
				memset( RN_graph_list_v, 0, sizeof( RN_graph_list_t));
				
				RN_graph_list_v->id = id2;
				RN_graph_list_v->leng = calc_leng( &RN_graph_v_ori->head_v[ id1].loc_v,
													&RN_graph_v_ori->head_v[ id2].loc_v);
				RN_graph_list_v->next = RN_graph_v_sim->head_v[ id1].list->next;
				RN_graph_v_sim->head_v[ id1].list->next = RN_graph_list_v;

				//2. id2->id1.
				RN_graph_list_v = ( RN_graph_list_t*)malloc( sizeof( RN_graph_list_t));
				memset( RN_graph_list_v, 0, sizeof( RN_graph_list_t));
				
				RN_graph_list_v->id = id1;
				RN_graph_list_v->leng = RN_graph_v_sim->head_v[ id1].list->next->leng;
				RN_graph_list_v->next = RN_graph_v_sim->head_v[ id2].list->next;
				RN_graph_v_sim->head_v[ id2].list->next = RN_graph_list_v;

				cur_t = pre_t;
				pre_t = pre[ cur_t];

				/*s*/
				RN_graph_v_sim->m_size += sizeof( RN_graph_list_t) * 2;
				/*s*/
			}
			
			break;
		}
	}//while( sta < rear)

	/*s*/
	stat_v.memory_v -= sizeof( int) * vertex_n * 3;
	/*s*/

	free( que);
	free( level);
	free( pre);
	
	return rtn;
}

/*
 *	Simplify a road network.
 *
 *	
 */
RN_graph_t* RN_simplify( RN_graph_t* RN_graph_v, float eps, int opt)
{
	int seg_n, vertex_n, rtn, edge_cnt, vertex_cnt;

	RN_graph_t* RN_graph_v_sim;
	R_set_t* R_set_v;
	R_list_t* R_list_v;
	eps_g_t* eps_g_v;

	//Alloc the memory for the new road network.
	vertex_n = RN_graph_v->vertex_n;
	RN_graph_v_sim = RN_graph_alloc( vertex_n);

	//Decompose the original graph.
	R_set_v = RN_decompose( RN_graph_v);

	edge_cnt = 0;
	vertex_cnt = RN_graph_v->vertex_n;
	R_list_v = R_set_v->head->next;
	while( R_list_v != NULL)
	{
		seg_n = R_list_v->size;
		
		//Construct the "G_{\epsilon_t}" graph.
		eps_g_v = eps_g_const( R_list_v, eps, opt);
		
		//Calculate the shortest path from vertex identified by 0 to that identified by "seg_n".
		rtn = calc_SP_RN( RN_graph_v_sim, RN_graph_v, eps_g_v, 0, seg_n);

		edge_cnt += rtn;
		vertex_cnt -= ( R_list_v->size - rtn);
		
		//Release the related resources.
		eps_g_release( eps_g_v);

		R_list_v = R_list_v->next;
	}

	RN_graph_v_sim->edge_n = edge_cnt;
	RN_graph_v_sim->vertex_n_sim = vertex_cnt;

	/*s*/
	stat_v.memory_v += RN_graph_v_sim->m_size;
	if( stat_v.memory_max < stat_v.memory_v)
		stat_v.memory_max = stat_v.memory_v;
	/*s*/

	//Release the resources.
	R_set_release( R_set_v);

	return RN_graph_v_sim;
}

/*
 *	The implementation of dijkstra.
 */
float dijkstra( RN_graph_t* RN_graph_v, int s_id, int d_id)
{
	return 0;
}

/*
 *	Verify the effectiveness of RN_simply.
 *
 *	@RN_graph_v_ori indicates the original road network.
 *	@RN_graph_v_sim indicates the simplified road network.
 *	@q_times indicates the times of shortest path query.
 *
 *	@The statistics information would be stored in stat_v.
 */
void RN_simplify_verify( RN_graph_t* RN_graph_v_ori, RN_graph_t* RN_graph_v_sim, int q_times)
{
	/*
	int i, s_id, d_id;

	for( i=0; i<q_times; i++)
	{
		//Specify the input of shortest path query: source & destination.
		//s_id, d_id.
		
		//Query on the original road network.

		//Query on the simplified road network.

		//Collect the statistics.
	}
	*/
}

/*
 *	Testing all the implementations [empirical study].
 */
void test2( )
{
	int num_res;
	float err_res;
	config_t* config_v;
	RN_graph_t* RN_graph_v;

	FILE* s_fp;

	memset( &stat_v, 0, sizeof( stat_t));
	stat_v.C.min = FLT_MAX;
	stat_v.C.max = 0;

	//Read the config.
	config_v = read_config( );

	//Read the road.
	RN_graph_v = read_data( config_v);

	/*t/
	print_RN_graph( RN_graph_v);
	/*t*/

	//Perform the calculation.
	num_res = 0;
	err_res = 0;

#ifndef WIN32
	struct rusage sta_t, end_t;
	get_cur_time( &sta_t);
#endif

	if( config_v->pro_opt == 1)		//Min-Number problem.
	{
		printf( "Min-Number:\n");
		if( config_v->alg_opt == 1)
			printf( "SP-Method\n\n");
		else
			printf( "DP-Number\n\n");
		num_res = Min_Number( RN_graph_v, config_v->eps, config_v->alg_opt);
		printf( "# of edges (original): %i\n", RN_graph_v->edge_n);
		printf( "Error tolerance      : %f\n", config_v->eps);
		printf( "# of edges (simplied): %i\n", num_res);
	}
	else 							//Min-Error problem.
	{
		printf( "Min-Error:\n");
		err_res = Min_Error( RN_graph_v, config_v->L, config_v->alg_opt);
		printf( "Quota L(%c): %f\n", 37, config_v->L_percent);
		printf( "Error:       %f\n", err_res);
	}

#ifndef WIN32
	get_cur_time( &end_t);
	collect_time( &sta_t, &end_t, &stat_v.r_time);
#endif

	//Print the statistics.
	if( ( s_fp = fopen( STATISTIC_FILE, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open %s\n", STATISTIC_FILE);
		exit( 0);
	}

	//if( config_v->pro_opt == 1)
		fprintf( s_fp, "%i\n", num_res);
	//else
		fprintf( s_fp, "%f\n\n", err_res);
	fprintf( s_fp, "%f\n%f\n\n", stat_v.r_time, stat_v.memory_max);
	fprintf( s_fp, "%f\n%f\n%f\n", stat_v.C.aver, stat_v.C.max, stat_v.C.min);

	//Release the related resources.
	free( config_v);
	//RN_graph_release( RN_graph_v);
	fclose( s_fp);

	stat_v.memory_v -= RN_graph_v->m_size;
	printf( "Memory: %f\n", stat_v.memory_v);
}

/*
 *	Test the implementation of RN_simplify [Empirical study].
 */
void test_RN_simplify( )
{
	config_t* config_v;
	RN_graph_t* RN_graph_v;
	RN_graph_t* RN_graph_v_sim;

	FILE* s_fp;

	memset( &stat_v, 0, sizeof( stat_t)); 
	stat_v.length_err.min = FLT_MAX;
	stat_v.length_err.max = 0;

	//Read the config.
	config_v = read_config( );

	//Read the road.
	RN_graph_v = read_data( config_v);

	//Perform the simplification procedure.
#ifndef WIN32
	struct rusage sta_t, end_t;
	get_cur_time( &sta_t);
#endif

	RN_graph_v_sim = RN_simplify( RN_graph_v, config_v->eps, 2);

	/*t*/
	printf( "seg_before simplification:\n");
	print_RN_graph( RN_graph_v);

	printf( "seg_after simplification:\n");
	print_RN_graph( RN_graph_v_sim);
	/*t*/

#ifndef WIN32
	get_cur_time( &end_t);
	collect_time( &sta_t, &end_t, &stat_v.r_time);
#endif

	//Verify the approximation road network.
	//print_RN_graph

	//Print the statistics.
	if( ( s_fp = fopen( STATISTIC_FILE, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open %s\n", STATISTIC_FILE);
		exit( 0);
	}

	fprintf( s_fp, "%f\n%f\n", stat_v.r_time, stat_v.memory_max);
	fprintf( s_fp, "%i\n", stat_v.C);

	//Release the related resources.
	free( config_v);
	RN_graph_release( RN_graph_v);
	fclose( s_fp);

	printf( "Memory: %f\n", stat_v.memory_v);
}




//Updated on 21 January 2013 for Min-Number.

/*
 *  Compute the fdr @slope_r_v of a segment @R_node_v.
 */
void calc_fdr_segment( slope_r_t* slope_r_v, R_node_t* R_node_v, float eps)
{
	float slope_v;

	slope_v = calc_abs_slope( R_node_v->loc_v1, R_node_v->loc_v2);
	slope_r_v->s_sta = modulo_op( slope_v - eps, 2 * PI);
	slope_r_v->s_end = modulo_op( slope_v + eps, 2 * PI);

	return;
}

/*
 *	Alloc an id_list_t struct
 */
id_list_t* id_list_alloc( )
{
	id_list_t* id_list_v;

	id_list_v = ( id_list_t*)malloc( sizeof( id_list_t));
	memset( id_list_v, 0, sizeof( id_list_t));

	id_list_v->head = ( id_node_t*)malloc( sizeof( id_node_t));
	memset( id_list_v->head, 0, sizeof( id_node_t));

	id_list_v->rear = id_list_v->head;

	/*s*/
	emp_stat_v.memory_v += sizeof( id_list_t) + sizeof( id_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return id_list_v;
}

/*
 *	Append a new id_node entry with @id to a id_list_t struct @id_list_v.
 */
void append_id_list_entry( id_list_t* id_list_v, int pos_id)
{
	id_node_t* id_node_v;

	id_node_v = ( id_node_t*)malloc( sizeof( id_node_t));
	memset( id_node_v, 0, sizeof( id_node_t));

	id_node_v->pos_id = pos_id;

	id_list_v->rear->next = id_node_v;
	id_list_v->rear = id_node_v;
	id_list_v->pos_id_n ++;

	/*s*/
	emp_stat_v.memory_v += sizeof( id_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return;
}

/*
 *	Add a new id_node ahead of a id_list_t struct.
 */
void ahead_id_list_entry( id_list_t* id_list_v, int pos_id)
{
	id_node_t* id_node_v;

	id_node_v = ( id_node_t*)malloc( sizeof( id_node_t));
	memset( id_node_v, 0, sizeof( id_node_t));

	id_node_v->pos_id = pos_id;

	id_node_v->next = id_list_v->head->next;
	id_list_v->head->next = id_node_v;
	if( id_list_v->rear == id_list_v->head)
		id_list_v->rear = id_node_v;

	id_list_v->pos_id_n ++;

	/*s*/
	emp_stat_v.memory_v += sizeof( id_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return;
}

/*
 *	Print a id_list_t struct.
 */
void id_list_print( id_list_t* id_list_v)
{
	id_node_t* iter;

	if( id_list_v == NULL)
	{
		printf( "Empty list.\n");
		return;
	}

	iter = id_list_v->head->next;
	while( iter != NULL)
	{
		printf( "%i  ", iter->pos_id);

		iter = iter->next;
	}
	printf( "\n");

	return;
}

/*
 *	Release an id_list_t struct.
 */
void id_list_release( id_list_t* id_list_v)
{
	id_node_t* id_node_v1, *id_node_v2;

	if( id_list_v == NULL)
		return;

	id_node_v1 = id_list_v->head;
	id_node_v2 = id_node_v1->next;
	while( id_node_v2 != NULL)
	{
		free( id_node_v1);

		id_node_v1 = id_node_v2;
		id_node_v2 = id_node_v1->next;
	}
	free( id_node_v1);

	/*s*/
	emp_stat_v.memory_v -= ( id_list_v->pos_id_n + 1) * sizeof( id_node_t) + sizeof( id_list_t);
	/*s*/

	free( id_list_v);
}

/*
 *	Same interface as calc_SP2 except the return argument.
 *
 *	Return the list of the id's corresponding the simplification.
 *
 */
id_list_t* calc_SP2( eps_g_t* eps_g_v, int s_id, int d_id)
{
	int i, vertex_n, sta, rear, c_id, n_id, rtn, cur_t;
	int* que, *level, *pre;

	eps_g_node_t* g_node_v;

	id_list_t* id_list_v;

	vertex_n = eps_g_v->vertex_n;
	que = ( int*)malloc( sizeof( int) * vertex_n);
	level = ( int*)malloc( sizeof( int) * vertex_n);
	pre = ( int*)malloc( sizeof( int) * vertex_n);
	memset( que, 0, sizeof( int) * vertex_n);
	memset( level, 0, sizeof( int) * vertex_n);
	memset( pre, 0, sizeof( int) * vertex_n);

	/*s*/
	emp_stat_v.memory_v += sizeof( int) * vertex_n * 3;
	if( emp_stat_v.memory_max < emp_stat_v.memory_v)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/
	
	for( i=0; i<vertex_n; i++)
		pre[ i] = -2;

	rtn = 0;

	sta = rear = -1;
	que[ ++rear] = s_id;
	pre[ s_id] = -1; 
	while( sta < rear)
	{
		c_id = que[ ++sta];

		g_node_v = eps_g_v->head[ c_id].list->next;
		while( g_node_v != NULL)
		{
			n_id = g_node_v->id;
			if( n_id == d_id)
			{
				//The shortest path found.
				rtn = level[ c_id] + 1;
				pre[ n_id] = c_id;

				break;
			}

			if( pre[ n_id] == -2)
			{
				/*t*/
				if( rear == vertex_n - 1)
				{
					//The destination must be reachable.
					fprintf( stderr, "Incorrect eps_g_v [calc_SP_RN].\n");
					exit( 0);
				}
				/*t*/

				que[ ++rear] = n_id;
				level[ n_id] = level[ c_id] + 1;
				pre[ n_id] = c_id;
			}
			
			g_node_v = g_node_v->next;
		}//while( g_node_v)


		if( rtn != 0)
		{
			//The shortest path has been found.
			//Trace the shortest path.
			id_list_v = id_list_alloc( );

			append_id_list_entry( id_list_v, d_id);
			

			cur_t = pre[ d_id];
			while( cur_t != -1)
			{	
				append_id_list_entry( id_list_v, cur_t);

				cur_t = pre[ cur_t];
			}
			
			break;
		}
	}//while( sta < rear)


	/*s*/
	emp_stat_v.memory_v -= sizeof( int) * vertex_n * 3;
	/*s*/

	free( que);
	free( level);
	free( pre);
	
	return id_list_v;
}

/*
 *	Construct a tra_list_t struct which is simplification of @ori_tra_list_v and 
 *	is indicated by a list of id's @id_list_v.
 */
tra_list_t* tra_const_from_id_list( tra_list_t* ori_tra_list_v, id_list_t* id_list_v)
{
	tra_list_t* simp_tra_list_v;
	tra_node_t* tra_node_v;

	id_node_t* id_node_v;

	simp_tra_list_v = tra_list_alloc( );
	

	id_node_v = id_list_v->head->next;

	tra_node_v = ori_tra_list_v->head->next;
	while( tra_node_v != NULL)
	{
		if( tra_node_v->pos_id == id_node_v->pos_id)
		{
			append_tra_list_entry( simp_tra_list_v, tra_node_v->tri_v);

			id_node_v = id_node_v->next;
		}
		
		tra_node_v = tra_node_v->next;
	}
	
	return simp_tra_list_v;
}


/*
 *	Allocate a fdr_array_t struct.
 */
fdr_array_t* fdr_array_alloc( int seg_n)
{
	int i;
	
	fdr_array_t* fdr_array_v;

	fdr_array_v = ( fdr_array_t*)malloc( sizeof( fdr_array_t));
	memset( fdr_array_v, 0, sizeof( fdr_array_t));

	fdr_array_v->seg_n = seg_n;

	fdr_array_v->fdr = ( slope_r_list_t***)malloc( seg_n * sizeof( slope_r_list_t**));
	memset( fdr_array_v->fdr, 0, seg_n * sizeof( slope_r_list_t**));

	fdr_array_v->rear_i = ( int*)malloc( seg_n * sizeof( int));
	memset( fdr_array_v->rear_i, 0, seg_n * sizeof( int));
	
	for( i=0; i<seg_n; i++)
		fdr_array_v->rear_i[ i] = i - 1;	//

	for( i=0; i<seg_n; i++)
	{
		fdr_array_v->fdr[ i] = ( slope_r_list_t**)malloc( seg_n * sizeof( slope_r_list_t*));
		memset( fdr_array_v->fdr[ i], 0, seg_n * sizeof( slope_r_list_t*));
	}

	/*s*/
	emp_stat_v.memory_v += sizeof( fdr_array_t) + seg_n * sizeof( slope_r_list_t**)
							+ seg_n * sizeof( int) + seg_n * seg_n * sizeof( slope_r_list_t*);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return fdr_array_v;
}


/*
 *	Release a fdr_array_t struct.
 */
void fdr_array_release( fdr_array_t* fdr_array_v)
{
	int i, j, seg_n;

	seg_n = fdr_array_v->seg_n;
	
	for( i=0; i<seg_n; i++)
	{
		for( j=0; j<seg_n; j++)
			slope_r_list_release( fdr_array_v->fdr[ i][ j]);

		free( fdr_array_v->fdr[ i]);
	}

	free( fdr_array_v->fdr);
	free( fdr_array_v->rear_i);


	/*s*/
	emp_stat_v.memory_v -= seg_n * seg_n * sizeof( slope_r_list_t*) + seg_n * sizeof( slope_r_list_t**)
						+ seg_n * sizeof( int) + sizeof( fdr_array_t);
	/*s*/

	free( fdr_array_v);
}

/*
 *	Maintain the fdr information.
 */
fdr_array_t* fdr_maintenance( R_list_t* R_list_v, float eps)
{
	int i, j, seg_n;
	float slope_v;

	fdr_array_t* fdr_array_v;
	slope_r_list_t*** fdr;

	slope_r_t slope_r_v;

	R_node_t* R_node_v1, *R_node_v2;


	seg_n = R_list_v->size;

	fdr_array_v = fdr_array_alloc( seg_n);
	fdr = fdr_array_v->fdr;

	R_node_v1 = R_list_v->head->next;
	for( i=0; i<seg_n; i++)
	{	
		//The fdr of the first segment.
		slope_v = R_node_v1->slope;
		slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
		slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

		//Maintain the slope range list.
		fdr[ i][ i] = slope_r_list_alloc( );
		append_slope_r_list_entry( fdr[ i][ i], slope_r_v);

		R_node_v2 = R_node_v1->next;
		for( j=i+1; j<seg_n; j++)
		{
			slope_v = R_node_v2->slope;
			slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
			slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

			fdr[ i][ j] = range_intersect( fdr[ i][ j-1], &slope_r_v);

			if( fdr[ i][ j] == NULL)
				break;							//Pay much attention!!!

			R_node_v2 = R_node_v2->next;
		}//for( j)

	}//for( i)

	return fdr_array_v;
}

/*
 *	Allocate a R_array_t struct.
 */
R_array_t* R_array_alloc( int seg_n)
{
	R_array_t* R_array_v;

	R_array_v = ( R_array_t*)malloc( sizeof( R_array_t));
	memset( R_array_v, 0, sizeof( R_array_t));

	R_array_v->size = seg_n;
	R_array_v->array = ( R_node_t**)malloc( seg_n * sizeof( R_node_t*));
	memset( R_array_v->array, 0, seg_n * sizeof( R_node_t*));

	/*s*/
	emp_stat_v.memory_v += sizeof( R_array_t) + seg_n * sizeof( R_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return R_array_v;
}

/*	
 *	Release a R_array_t struct.
 */	
void R_array_release( R_array_t* R_array_v)
{
	/*s*/
	emp_stat_v.memory_v -= R_array_v->size * sizeof( R_node_t) + sizeof( R_array_t);
	/*s*/

	free( R_array_v->array);
	free( R_array_v);
}

/*
 *	Transform a R_list_t struct to a R_array_t one.
 */
R_array_t* R_array_transform( R_list_t* R_list_v)
{
	int i;

	R_array_t* R_array_v;
	R_node_t* R_node_v;

	R_array_v = R_array_alloc( R_list_v->size);


	i = 0;
	R_node_v = R_list_v->head->next;
	while( R_node_v != NULL)
	{
		R_array_v->array[ i++] = R_node_v;

		R_node_v = R_node_v->next;
	}

	return R_array_v;
}

/*
 *	Incrementally maintain the fdr array.
 *	Maintain fdr[ sta][ end] if it has not been maintained.
 *
 */
void fdr_maintenance_on_demand( fdr_array_t* fdr_array_v, R_array_t* R_array_v, float eps, int seg_sta, int seg_end)
{
	int j;
	float slope_v;

	slope_r_t slope_r_v;


	if( seg_end <= fdr_array_v->rear_i[ seg_sta])
		return;
	
	for( j=fdr_array_v->rear_i[ seg_sta]+1; j<=seg_end; j++)
	{
		//Compute fdr( T[i, j]);
		//fdr( \overline{p_{j-1}p_j});
		slope_v = R_array_v->array[ j]->slope;
		slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
		slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);

		/*t*/
		emp_stat_v.eps_n ++;
		/*t*/

		//
		if( j == seg_sta)
		{
			fdr_array_v->fdr[ seg_sta][ j] = slope_r_list_alloc( );
			append_slope_r_list_entry( fdr_array_v->fdr[ seg_sta][ j], slope_r_v);
			
			continue;
		}

		//fdr( T[i, j]) = fdr( T[i, j-1) \cap fdr(\overline{p_{j-1}p_j});
		fdr_array_v->fdr[ seg_sta][ j] = range_intersect( fdr_array_v->fdr[ seg_sta][ j-1], &slope_r_v);
		
		if( fdr_array_v->fdr[ seg_sta][ j] == NULL)
		{
			fdr_array_v->rear_i[ seg_sta] = R_array_v->size - 1;
			return;
		}
	}

	fdr_array_v->rear_i[ seg_sta] = seg_end;
}

/*
 *	Check the error with the fdr information.
 */	
int check_error_fdr( fdr_array_t* fdr_array_v, R_array_t* R_array_v, float eps, int pos_sta, int pos_end)
{
	float slope;

	//Maintain the fdr informaiton.
	fdr_maintenance_on_demand( fdr_array_v, R_array_v, eps, pos_sta, pos_end-1);

	//Compute the direction of $\overline{p_{sta}p_{end}}$.
	slope = calc_abs_slope( R_array_v->array[ pos_sta]->loc_v1, R_array_v->array[ pos_end-1]->loc_v2);

	if( is_in_slope_r_list( fdr_array_v->fdr[ pos_sta][ pos_end-1], slope))
		return 1;
	else
		return 0;
}

/*
 *	Alloc a hop_list_t struct.
 */
hop_list_t* hop_list_alloc( )
{
	hop_list_t* hop_list_v;

	hop_list_v = ( hop_list_t*)malloc( sizeof( hop_list_t));
	memset( hop_list_v, 0, sizeof( hop_list_t));

	hop_list_v->head = ( hop_node_t*)malloc( sizeof( hop_node_t));
	memset( hop_list_v->head, 0, sizeof( hop_node_t));


	hop_list_v->rear = hop_list_v->head;

	/*s*/
	emp_stat_v.memory_v += sizeof( hop_list_t) + sizeof( hop_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return hop_list_v;
}

/*
 *	Add a new hop_list_t entry ahead of the list.
 */
void ahead_hop_list_entry( hop_list_t* hop_list_v, int pos_id, R_node_t* seg_before, R_node_t* seg_after)
{
	hop_node_t* hop_node_v;

	hop_node_v = ( hop_node_t*)malloc( sizeof( hop_node_t));
	memset( hop_node_v, 0, sizeof( hop_node_t));

	hop_node_v->pos_id = pos_id;
	//hop_node_v->pre_id = pre_id;
	hop_node_v->seg_before = seg_before;
	hop_node_v->seg_after = seg_after;
	
	//hop_list_v->rear->next = hop_node_v;
	hop_node_v->next = hop_list_v->head->next;
	if( hop_node_v->next != NULL)
		hop_node_v->next->pre = hop_node_v;
	hop_node_v->pre = hop_list_v->head;
	hop_list_v->head->next = hop_node_v;

	if( hop_list_v->rear == hop_list_v->head)
		hop_list_v->rear = hop_node_v;

	/*s*/
	emp_stat_v.memory_v += sizeof( hop_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return;
}

/*
 *	Append a hop_list_t entry.
 */
void append_hop_list_entry( hop_list_t* hop_list_v, int pos_id, R_node_t* seg_before, R_node_t* seg_after)
{
	hop_node_t* hop_node_v;

	hop_node_v = ( hop_node_t*)malloc( sizeof( hop_node_t));
	memset( hop_node_v, 0, sizeof( hop_node_t));

	hop_node_v->pos_id = pos_id;
	//hop_node_v->pre_id = pre_id;
	hop_node_v->seg_before = seg_before;
	hop_node_v->seg_after = seg_after;
	
	hop_node_v->pre = hop_list_v->rear;
	hop_list_v->rear->next = hop_node_v;
	hop_list_v->rear = hop_node_v;

	/*s*/
	emp_stat_v.memory_v += sizeof( hop_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return;
}

/*
 *	Release a hop_list_t struct.
 */
void hop_list_release( hop_list_t* hop_list_v)
{
	hop_list_t* next_hop_list_v;
	hop_node_t* hop_node_v1, *hop_node_v2;

	while( hop_list_v != NULL)
	{
		next_hop_list_v = hop_list_v->next;

		//release hope_list_v.
		hop_node_v1 = hop_list_v->head;
		while( hop_node_v1 != NULL)
		{
			hop_node_v2 = hop_node_v1->next;

			free( hop_node_v1);

			/*s*/
			emp_stat_v.memory_v -= sizeof( hop_node_t);
			/*s*/

			hop_node_v1 = hop_node_v2;
		}

		free( hop_list_v);

		/*s*/
		emp_stat_v.memory_v -= sizeof( hop_list_t);
		/*s*/

		hop_list_v = next_hop_list_v;
	}

	return;
}


/*
 *	The implementation of the SP-Method-E2 algorithm
 *	which returns the size of the optimal simplification.
 *
int SP_Method_prac_num( R_list_t* R_list_v, float eps, int fdr_opt)
{
	return 0;
}*/


/*
 *	The implementation of the SP method with the practical enhancement.
 */
id_list_t* SP_Method_prac( R_list_t* R_list_v, float eps)
{
	int seg_n, pos_n, pos_id1, pos_id2, error_i, i, cur_id, cnt;
	int *pre;
	
	hop_list_t* H, *U, *hop_list_v, *pre_hop_list_v;
	hop_node_t* hop_node_v1, *hop_node_v2, *hop_node_v3;
	
	R_node_t* seg_before, *seg_after;
	
	id_list_t* id_list_v;
	
	
	seg_n = R_list_v->size;
	pos_n = seg_n + 1;
	
	pre = ( int*)malloc( pos_n * sizeof( int));
	memset( pre, 0, pos_n * sizeof( int));
	
	/*s*/
	emp_stat_v.memory_v += pos_n * sizeof( int);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/
	
	//Initialize H_0
	H = hop_list_alloc( );	
	seg_before = NULL;
	seg_after = R_list_v->head->next;
	append_hop_list_entry( H, 0, seg_before, seg_after);
	
	pre[ 0] = -1;
	
	
	//Initialize U.
	U = hop_list_alloc( );
	for( i=1; i<pos_n; i++)
	{
		seg_before = seg_after;
		seg_after = seg_before->next;
		//append_hop_list_entry( U, i, seg_before, seg_after);
		
		ahead_hop_list_entry( U, i, seg_before, seg_after);
	}
	
	
	//Compute H_h's for h=1, 2, ...
	pre_hop_list_v = H;
	cnt = 0;
	while( true)
	{
		/*t/
		printf( "%i-length set: \n", cnt ++);
		//print_hop_list( pre_hop_list_v);
		/*t*/
		
		hop_list_v = hop_list_alloc( );
		pre_hop_list_v->next = hop_list_v;
		
		hop_node_v1 = pre_hop_list_v->head->next;
		while( hop_node_v1 != NULL)
		{			
			pos_id1 = hop_node_v1->pos_id;

			/*t/
			printf( "U: ");
			print_hop_list( U);
			/*t*/
			
			
			//hop_node_v3 = U->head;
			hop_node_v2 = U->head->next;
			while( hop_node_v2 != NULL)
			{
				if( hop_node_v2->pos_id <= hop_node_v1->pos_id)
					break;			//
				
				pos_id2 = hop_node_v2->pos_id;
				
				//Compute the error.
				
				//Option 1: using the straightforward method (O(n)).
				error_i = check_error( hop_node_v1->seg_after, hop_node_v2->seg_before, eps);
				
				/*s*/
				emp_stat_v.eps_n ++;
				/*s*/
				
				
				if( error_i)
				{
					if( pos_id2 == pos_n - 1)
					{
						//The shortest path has been founded.
						//Check out the shortest path.
						pre[ pos_id2] = pos_id1;
						
						id_list_v = id_list_alloc( );
						
						cur_id = pos_id2;
						//append_id_list_entry( id_list_v, cur_id);
						ahead_id_list_entry( id_list_v, cur_id);
						while( pre[ cur_id] != -1)
						{
							/*t/
							if( pre[ cur_id] < 0)
								printf( "");
							/*t*/
							
							cur_id = pre[ cur_id];
							
							ahead_id_list_entry( id_list_v, cur_id);
						}
						
						//Release the hop_list_t struct.
						hop_list_release( H);
						hop_list_release( U);
						
						
						free( pre);
						
						/*s*/
						emp_stat_v.memory_v -= pos_n * sizeof( int);
						/*s*/
						
						return id_list_v;
					}
					
					//Remove the position with id2 from U;
					hop_node_v3 = hop_node_v2->pre;
					hop_node_v3->next = hop_node_v2->next;
					if( hop_node_v2->next != NULL)
					{
						hop_node_v2->next->pre = hop_node_v3;
					}
					
					if( U->rear == hop_node_v2)
						U->rear = hop_node_v3;
					
					//Add the position with id2 into H_h;
					hop_node_v2->next = NULL;
					hop_node_v2->pre = hop_list_v->rear;
					hop_list_v->rear->next = hop_node_v2;
					hop_list_v->rear = hop_node_v2;
					
					pre[ pos_id2] = pos_id1;				
					
					hop_node_v2 = hop_node_v3;
				}//if( error_i)
				
				hop_node_v2 = hop_node_v2->next;							
			}//while( hop_node_v2)
			
			
			hop_node_v1 = hop_node_v1->next;
			
		}//while( hop_node_v1)
		
		pre_hop_list_v = hop_list_v;		
	}//while(true)
	
	/*t*/
	fprintf( stderr, "Bug in [SP_Method_sub2]...\n");
	exit( 0);
	/*t*/
}

/*
 *	The implementation of the SP-Method-E2 algorithm
 *	which returns the optimal simplification exactly.
 */
id_list_t* SP_Method_prac_and_theo( R_list_t* R_list_v, float eps)
{
	int seg_n, pos_n, seg_left_n, pos_id1, pos_id2, error_i, i, j, k, cur_id, cnt, seg_ind, cur_seg_id, tag, tag2;
	int *pre;
	float slope_v;
	
	hop_list_t* H, *U, *hop_list_v, *pre_hop_list_v;
	hop_node_t* hop_node_v1, *hop_node_v2, *hop_node_v3, *hop_node_v4;
	
	R_node_t* seg_before, *seg_after;
	
	slope_r_t slope_r_v;
	
	//fdr_array_t* fdr_array_v;
	slope_r_list_t** fdr_v;
	R_array_t* R_array_v;
	
	id_list_t* id_list_v;
	
	R_array_v = R_array_transform( R_list_v);
	
	
	seg_n = R_list_v->size;
	pos_n = seg_n + 1;
	
	pre = ( int*)malloc( pos_n * sizeof( int));
	memset( pre, 0, pos_n * sizeof( int));
	
	/*s*/
	emp_stat_v.memory_v += pos_n * sizeof( int);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/
	
	//Initialize H_0
	H = hop_list_alloc( );	
	seg_before = NULL;
	seg_after = R_list_v->head->next;
	append_hop_list_entry( H, 0, seg_before, seg_after);
	
	pre[ 0] = -1;
	
	
	//Initialize U.
	U = hop_list_alloc( );
	for( i=1; i<pos_n; i++)
	{
		seg_before = seg_after;
		seg_after = seg_before->next;
		//append_hop_list_entry( U, i, seg_before, seg_after);
		
		ahead_hop_list_entry( U, i, seg_before, seg_after);
	}
	
	
	//Compute H_h's for h=1, 2, ...
	pre_hop_list_v = H;
	cnt = 0;
	while( true)
	{
		/*t/
		printf( "%i-length set: \n", cnt ++);
		//print_hop_list( pre_hop_list_v);
		/*t*/
		
		hop_list_v = hop_list_alloc( );
		pre_hop_list_v->next = hop_list_v;
		
		hop_node_v1 = pre_hop_list_v->head->next;
		while( hop_node_v1 != NULL)
		{			
			/*t/
			printf( "U: ");
			print_hop_list( U);
			/*t*/
			
			//hop_node_v1 is only valid in one specific iteration.
			//Maintain the fdr information locally and on-demand.
					
			pos_id1 = hop_node_v1->pos_id;
			seg_left_n = pos_n - 1 - pos_id1;
			
			fdr_v = ( slope_r_list_t**)malloc( seg_left_n * sizeof( slope_r_list_t*));
			memset( fdr_v, 0, seg_left_n * sizeof( slope_r_list_t*));
			
			tag = 0;
			tag2 = 0;
			
			
			/*s*/
			emp_stat_v.memory_v += seg_left_n * sizeof( slope_r_list_t*);
			if( emp_stat_v.memory_v > emp_stat_v.memory_max)
				emp_stat_v.memory_max = emp_stat_v.memory_v;
			/*s*/
			
			
			//hop_node_v3 = U->head;
			hop_node_v2 = U->head->next;
			while( hop_node_v2 != NULL)
			{
				if( hop_node_v2->pos_id <= hop_node_v1->pos_id)
					break;			//
				
				pos_id2 = hop_node_v2->pos_id;			
				
				//Maintain the fdr information on-demand.
				if( tag == 0)
				{
					//
					tag = 1;
					
					for( j=pos_id1+1; j<=pos_id2; j++)
					{
						/*s*/
						emp_stat_v.eps_n ++;
						/*s*/
						
						seg_ind = j - 1 - pos_id1;
						cur_seg_id = j - 1;
						
						slope_v = calc_abs_slope( R_array_v->array[ cur_seg_id]->loc_v1, 
							R_array_v->array[ cur_seg_id]->loc_v2);
						slope_r_v.s_sta = modulo_op( slope_v - eps, 2 * PI);
						slope_r_v.s_end = modulo_op( slope_v + eps, 2 * PI);
						
						if( seg_ind == 0)
						{
							//The first fdr information.
							fdr_v[ seg_ind] = slope_r_list_alloc( );
							append_slope_r_list_entry( fdr_v[ seg_ind], slope_r_v);
						}
						else
						{
							fdr_v[ seg_ind] = range_intersect( fdr_v[ seg_ind - 1], &slope_r_v);
						}
						
						if( fdr_v[ seg_ind] == NULL)
						{
							//For efficiency consideration.
							//Re-set hop_node_v2 here.
							hop_node_v4 = U->rear;
							while( hop_node_v4 != NULL && hop_node_v4->pos_id < j - 1)
								hop_node_v4 = hop_node_v4->pre;
							
							/*t*/
							if( hop_node_v4 == NULL)
							{
								fprintf( stderr, "Bug [SP_Method_sub2]\n");
								exit( 0);
							}
							/*t*/
							
							if( hop_node_v4->pos_id > j)
								hop_node_v4 = hop_node_v4->pre;
							
							hop_node_v2 = hop_node_v4;
							
							
							tag2 = 1;
							break;
						}
					}//for( j)
				}//if( tag1 == 0)
				
				if( tag2 == 1)
				{
					tag2 = 0;
					continue;
				}
				
				//Check the error.
				//error_i = check_error_fdr( fdr_array_v, R_array_v, eps, pos_id1, pos_id2);
				if( fdr_v[ pos_id2 - 1 - pos_id1] == NULL)
					error_i = 0;
				else
				{
					if( pos_id2 == pos_id1 + 1)
						error_i = 1;
					else
					{
						slope_v = calc_abs_slope( R_array_v->array[ pos_id1]->loc_v1, R_array_v->array[ pos_id2-1]->loc_v2);
						if( is_in_slope_r_list( fdr_v[ pos_id2 - 1 - pos_id1], slope_v))
							error_i = 1;
						else
							error_i = 0;
					}
				}
				
				
				if( error_i)
				{
					if( pos_id2 == pos_n - 1)
					{
						//The shortest path has been founded.
						//Check out the shortest path.
						pre[ pos_id2] = pos_id1;
						
						id_list_v = id_list_alloc( );
						
						cur_id = pos_id2;
						//append_id_list_entry( id_list_v, cur_id);
						ahead_id_list_entry( id_list_v, cur_id);
						while( pre[ cur_id] != -1)
						{
							/*t/
							if( pre[ cur_id] < 0)
								printf( "");
							/*t*/
							
							cur_id = pre[ cur_id];
							
							ahead_id_list_entry( id_list_v, cur_id);
						}
						
						//Release the hop_list_t struct.
						hop_list_release( H);
						hop_list_release( U);
						
						
						R_array_release( R_array_v);
						//fdr_array_release( fdr_array_v);
						
						for( k=0; k<seg_left_n; k++)
							slope_r_list_release( fdr_v[ k]);
						
						free( fdr_v);
				
						free( pre);
						
						/*s*/
						emp_stat_v.memory_v -= pos_n * sizeof( int) + seg_left_n * sizeof( slope_r_list_t*);
						/*s*/
						
						return id_list_v;
					}
					
					//Remove the position with id2 from U;
					hop_node_v3 = hop_node_v2->pre;
					hop_node_v3->next = hop_node_v2->next;
					if( hop_node_v2->next != NULL)
					{
						hop_node_v2->next->pre = hop_node_v3;
					}
					
					if( U->rear == hop_node_v2)
						U->rear = hop_node_v3;
					
					//Add the position with id2 into H_h;
					hop_node_v2->next = NULL;
					hop_node_v2->pre = hop_list_v->rear;
					hop_list_v->rear->next = hop_node_v2;
					hop_list_v->rear = hop_node_v2;
					
					pre[ pos_id2] = pos_id1;				
					
					hop_node_v2 = hop_node_v3;
				}//if( error_i)
				
				hop_node_v2 = hop_node_v2->next;							
			}//while( hop_node_v2)
			
			
			for( k=0; k<seg_left_n; k++)
			{
				if( fdr_v[ k] == NULL)
					break;

				slope_r_list_release( fdr_v[ k]);
			}
			
			free( fdr_v);
			
			/*s*/
			emp_stat_v.memory_v -= seg_left_n * sizeof( slope_r_list_t*);
			/*s*/			
			
			hop_node_v1 = hop_node_v1->next;
			
		}//while( hop_node_v1)
		
		pre_hop_list_v = hop_list_v;		
	}//while(true)
	
	/*t*/
	fprintf( stderr, "Bug in [SP_Method_prac2]...\n");
	exit( 0);
	/*t*/
}

/*
 *	The basic version of SP.
 */
int SP_Method_B( R_list_t* R_list_v, float eps)
{
	return SP_Method_sub( R_list_v, eps, 0);
}

/*
 * The SP with the thorectical enhancement.
 */
int SP_Method_theo( R_list_t* R_list_v, float eps)
{
	return SP_Method_sub( R_list_v, eps, 1);
}

/*
 *	Input trajectory is T[seg_sta, seg_end+1].
 */
int Split_sub( R_array_t* R_array_v, int seg_sta, int seg_end, float eps)
{
	int k, split_seg, i, j;
	float error_max, angle_v;

	i = seg_sta;
	j = seg_end;

	if( i > j)
		return 0;

	if( i == j)
		return 2;

	//Compute the \epsilon(\overline{p_ip_j}).
	error_max = -1;
	split_seg = i;
	for( k=i; k<=j; k++)
	{
		//error_v = check_error( R_array_v->array[ k], &R_node_v);
		angle_v = calc_angle( R_array_v->array[ k]->loc_v1, R_array_v->array[ k]->loc_v2,
								R_array_v->array[ i]->loc_v1, R_array_v->array[ j]->loc_v2);

		if( angle_v > error_max)
		{
			error_max = angle_v;
			split_seg = k;
		}
	}

	if( error_max <= eps)
		return 2;

	if( split_seg == j)
		split_seg -= 1;

	return Split_sub( R_array_v, i, split_seg, eps) + Split_sub( R_array_v, split_seg + 1, j, eps);
}

//Approximate algorithms for the Min-Number problem.
/*
 *	The implementation of the Split algorithm.
 */
int Split( R_list_t* R_list_v, float eps)
{
	int simp_size;
	R_array_t* R_array_v;

	R_array_v = R_array_transform( R_list_v);

	simp_size = Split_sub( R_array_v, 0, R_list_v->size-1, eps);

	R_array_release( R_array_v);

	return simp_size;
}

/*
 *	Allocate a group_list_t struct.
 */
group_list_t* group_list_alloc( )
{
	group_list_t* group_list_v;

	group_list_v = ( group_list_t*)malloc( sizeof( group_list_t));
	memset( group_list_v, 0, sizeof( group_list_t));

	group_list_v->head = ( group_node_t*)malloc( sizeof( group_node_t));
	memset( group_list_v->head, 0, sizeof( group_node_t));

	group_list_v->rear = group_list_v->head;

	/*s*/
	emp_stat_v.memory_v += sizeof( group_list_t) + sizeof( group_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return group_list_v;
}

/*
 *	Append a group_node_t indicated by @sta_id and @end_id to a group_list_t struct.
 */
void append_group_list_entry( group_list_t* group_list_v, int sta_seg_id, int end_seg_id)
{
	group_node_t* group_node_v;

	group_node_v = ( group_node_t*)malloc( sizeof( group_node_t));
	memset( group_node_v, 0, sizeof( group_node_t));

	group_node_v->sta_seg_id = sta_seg_id;
	group_node_v->end_seg_id = end_seg_id;

	group_list_v->rear->next = group_node_v;
	group_node_v->pre = group_list_v->rear;
	group_list_v->rear = group_node_v;
	group_list_v->size ++;

	/*s*/
	emp_stat_v.memory_v += sizeof( group_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return;
}

/*
 *	Release a group_list_t struct.
 */
void group_list_release( group_list_t* group_list_v)
{
	group_node_t* group_node_v1, *group_node_v2;

	group_node_v1 = group_list_v->head;
	while( group_node_v1 != NULL)
	{
		group_node_v2 = group_node_v1->next;
		
		free( group_node_v1);
		group_node_v1 = group_node_v2;
	}

	/*s*/
	emp_stat_v.memory_v -= sizeof( group_node_t) * ( group_list_v->size + 1) + sizeof( group_list_t);
	/*s*/

	free( group_list_v);
	return;
}

/*
 *	The implementation of the Merge algorithm.
 *	The current implementation does not adopt the priority queue.
 *	
 */
id_list_t* Merge_sub( R_array_t* R_array_v, float eps)
{
	int i, sta, end;
	float error_min, error_v;

	group_node_t* merge_loc, *group_node_v1, *group_node_v2, *group_node_v;
	group_list_t* group_list_v;

	id_list_t* id_list_v;

	if( R_array_v->size == 0)
		return id_list_alloc( );
	
	//Initialize the group_list_t struct.
	group_list_v = group_list_alloc( );

	for( i=0; i<R_array_v->size; i++)
	{
		append_group_list_entry( group_list_v, i, i);
	}
	
	while( true)
	{
		//Finding the merge location.
		error_min = FLT_MAX;
		merge_loc = NULL;

		group_node_v1 = group_list_v->head->next;
		while( group_node_v1->next != NULL)
		{
			group_node_v2 = group_node_v1->next;

			//Compute the error of the merged group.
			sta = group_node_v1->sta_seg_id;
			end = group_node_v2->end_seg_id;

			error_v = calc_error( R_array_v->array[ sta], R_array_v->array[ end]);

			if( error_v <= eps && error_v < error_min)
			{
				error_min = error_v;
				merge_loc = group_node_v1;
			}

			group_node_v1 = group_node_v2;
		}

		if( merge_loc == NULL)
			break;

		//Perform the merge operation.
		group_node_v = merge_loc->next;

		merge_loc->end_seg_id = group_node_v->end_seg_id;
		merge_loc->next = group_node_v->next;
		group_list_v->size --;

		if( group_list_v->rear == group_node_v)
			group_list_v->rear = merge_loc;

		free( group_node_v);

		/*s*/
		emp_stat_v.memory_v -= sizeof( group_node_t);
		/*s*/
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

	return id_list_v;
}

/*
 *	The improved implementation of Merge_sub
 *	with the heap structure.
 *
 *	It is possible that the R_array_v structur could be replaced by the R_list_t structure.
 *
 *	Re-design!
 *	Avoid the use the group_list_t structure.
 */
id_list_t* Merge_sub2( R_array_t* R_array_v, float eps)
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

		if( b_h->obj_arr[ top].key > eps)
			break;

		
		//Perform the merge operation.
		group_node_v = b_h->obj_arr[ top].group_node_v;
		group_node_v1 = group_node_v->pre;
		group_node_v2 = group_node_v->next;

		/*t*/
		if( group_node_v2 == NULL)
		{
			fprintf( stderr, "bug. [Merge_sub2]\n");
			exit( 0);
		}
		/*t*/

		group_node_v1->next = group_node_v2;

		group_node_v2->sta_seg_id = group_node_v->sta_seg_id;

		group_node_v2->pre = group_node_v1;
		group_list_v->size --;

		free( group_node_v);

		/*s*/
		emp_stat_v.memory_v -= sizeof( group_node_t);
		/*s*/

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

		//
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
id_list_t* Merge( R_list_t* R_list_v, float eps)
{
	R_array_t* R_array_v;
	id_list_t* id_list_v;

	MIN_HEAP_OPT = 1;

	R_array_v = R_array_transform( R_list_v);

	//id_list_v = Merge_sub2( R_array_v, eps);
	id_list_v = Merge_sub2( R_array_v, eps);

	R_array_release( R_array_v);

	return id_list_v;
}

/*
 *	The implementation of the Greedy algorithm
 *	where $\epsilon(\overline{p_ip_j})$ is computed with a straightforward method (O(n)).
 */
id_list_t* Greedy_v1( R_list_t* R_list_v, float eps)
{
	id_list_t* id_list_v;
	R_node_t* R_node_v1, *R_node_v2;

	//R_node_v1 is the starting segment.
	//R_node_v2 is the current segment.


	R_node_v1 = R_list_v->head->next;

	id_list_v = id_list_alloc( );
	append_id_list_entry( id_list_v, R_node_v1->id1);

	R_node_v2 = R_node_v1->next;
	while( R_node_v2 != NULL)
	{
		if( !check_error( R_node_v1, R_node_v2, eps))
		{
			append_id_list_entry( id_list_v, R_node_v2->id1);

			R_node_v1 = R_node_v2;
		}

		R_node_v2 = R_node_v2->next;
	}

	append_id_list_entry( id_list_v, R_list_v->size);

	return id_list_v;
}

/*
 *	The implementation of the Greedy algorithm
 *	where $\epsilon(\overline{p_ip_j})$ is computed with the help of the "fdr" information.
 */
id_list_t* Greedy_v2( R_list_t* R_list_v, float eps)
{
	float slope_v;

	slope_r_t slope_r_v;
	slope_r_list* fdr, *fdr_t;
	R_node_t* R_node_v1, *R_node_v2;
	id_list_t* id_list_v;


	R_node_v1 = R_list_v->head->next;
	if( R_node_v1 == NULL)
		return id_list_alloc( );

	//Append the first position.
	id_list_v = id_list_alloc( );
	append_id_list_entry( id_list_v, R_node_v1->id1);

	//Initialize the fdr.
	fdr = slope_r_list_alloc( );

	calc_fdr_segment( &slope_r_v, R_node_v1, eps);

	append_slope_r_list_entry( fdr, slope_r_v);

	
	//R_node_v1 is the starting segment.
	//R_node_v2 is the current segment.
	R_node_v2 = R_node_v1->next;
	while( R_node_v2 != NULL)
	{		
		//Compute the fdr.
		calc_fdr_segment( &slope_r_v, R_node_v2, eps);

		fdr_t = range_intersect( fdr, &slope_r_v);
		slope_r_list_release( fdr);
		fdr = fdr_t;

		//Compute the slope.
		slope_v = calc_abs_slope( R_node_v1->loc_v1, R_node_v2->loc_v2);

		//Check the error.
		if( !is_in_slope_r_list( fdr, slope_v))
		{
			//Append the position.
			append_id_list_entry( id_list_v, R_node_v2->id1);

			//Re-set the starting segment.
			R_node_v1 = R_node_v2;

			//Re-initialize the fdr information.
			slope_r_list_release( fdr);
			fdr = slope_r_list_alloc( );

			calc_fdr_segment( &slope_r_v, R_node_v1, eps);
			append_slope_r_list_entry( fdr, slope_r_v);
		}

		R_node_v2 = R_node_v2->next;
	}

	//Append the last position.
	append_id_list_entry( id_list_v, R_list_v->rear->id2);

	//Release the resource.
	slope_r_list_release( fdr);

	return id_list_v;
}

/*	
 *	The implementation of the Intersect algorithm.
 */
id_list_t* Intersect( R_list_t* R_list_v, float eps)
{
	R_node_t* R_node_v1, *R_node_v2;
	slope_r_t slope_r_v;
	slope_r_list_t* fdr, *fdr_t;
	id_list_t* id_list_v;

	R_node_v1 = R_list_v->head->next;
	if( R_node_v1 == NULL)
		return id_list_alloc( );


	//Append the first position.
	id_list_v = id_list_alloc( );
	append_id_list_entry( id_list_v, R_node_v1->id1);


	//Initialize the fdr variable.
	fdr = slope_r_list_alloc( );

	calc_fdr_segment( &slope_r_v, R_node_v1, eps / 2);

	append_slope_r_list_entry( fdr, slope_r_v);

	/*t/
	print_slope_r_list( fdr);
	/*t*/


	//R_node_v1 is the starting segment.
	//R_node_v2 is the current segment.
	R_node_v2 = R_node_v1->next;
	while( R_node_v2 != NULL)
	{
		calc_fdr_segment( &slope_r_v, R_node_v2, eps / 2);

		fdr_t = range_intersect( fdr, &slope_r_v);
		slope_r_list_release( fdr);
		fdr = fdr_t;

		/*t/
		print_slope_r_list( fdr);
		/*t*/

		if( fdr == NULL)
		{
			//Append a position.
			append_id_list_entry( id_list_v, R_node_v2->id1);

			//Re-initialize fdr.
			fdr = slope_r_list_alloc( );
			//calc_fdr_segment( &slope_r_v, R_node_v2, eps / 2);
			append_slope_r_list_entry( fdr, slope_r_v);
		}

		R_node_v2 = R_node_v2->next;
	}

	//Append the last position.
	append_id_list_entry( id_list_v, R_list_v->rear->id2);

	//Release the resource.
	slope_r_list_release( fdr);

	return id_list_v;
}


//For collecting error statistics.
/*
 *	Collect the length and speed errors.
 */
void collect_length_and_speed_error( R_list_t* R_list_v_ori, R_list_t* R_list_v_appr, float* length_error, float* speed_error)
{
	float length_v, time_v, speed_v, length_ratio_v, speed_ratio_v, length_ratio_min, speed_ratio_min;

	R_node_t* R_node_v1, *R_node_v2;

	length_ratio_min = 1;
	speed_ratio_min = 1;

	R_node_v1 = R_list_v_appr->head->next;
	R_node_v2 = R_list_v_ori->head->next;
	while( R_node_v1 != NULL)
	{
		//Compute the length and speed error of segment R_node_v1;
		
		//t
		if( R_node_v2->id1 != R_node_v1->id1)
		{
			fprintf( stderr, "R_list_v_appr inconsistency [collect_length_and_speed_error].\n");
			exit( 0);
		}
		//t			

		length_v = 0;
		time_v = 0;
		while( R_node_v2 != NULL)
		{
			length_v += R_node_v2->leng;
			time_v += R_node_v2->t_interval;

			if( R_node_v2->id2 == R_node_v1->id2)
				break;

			R_node_v2 = R_node_v2->next;
		}

		if( R_node_v2 == NULL)
		{
			fprintf( stderr, "R_list_v_appr inconsistency [collect_length_and_speed_error].\n");
			exit( 0);
		}
		
		speed_v = length_v / time_v;

		length_ratio_v = R_node_v1->leng / length_v;
		speed_ratio_v = ( R_node_v1->leng / R_node_v1->t_interval) / speed_v;

		if( length_ratio_v < length_ratio_min)
			length_ratio_min = length_ratio_v;
		if( speed_ratio_v < speed_ratio_min)
			speed_ratio_min = speed_ratio_v;
		
		R_node_v1 = R_node_v1->next;
		R_node_v2 = R_node_v2->next;

		length_v = 0;
		time_v = 0;
	}//while
	
	*length_error = length_ratio_min;
	*speed_error = speed_ratio_min;

	return;
}

/*
 *	
 */
float collect_position_error_bound( R_list_t* R_list_v_appr, float eps, int dist_opt)
{
	DIST_TYPE length_max;
	R_node_t* iter;

	length_max = 0;
	iter = R_list_v_appr->head->next;
	while( iter != NULL)
	{
		if( iter->leng > length_max)
			length_max = iter->leng;

		iter = iter->next;
	}

	if( dist_opt == 1)
	{
		return ( float)( 0.5 * length_max * tan( eps));
	}
	else
	{
		return ( float)( 0.5 * length_max * ( 1 + 1 / cos( eps)));
	}
}

/*
 *	Collect the position errors.
 */
void collect_position_error( R_list_t* R_list_v_ori, R_list_t* R_list_v_appr, float* position_error, int dist_opt)
{
	float position_error_max, position_error_v, time_cnt;

	R_node_t* R_node_v1, *R_node_v2;

	position_error_max = 0;
	
	R_node_v1 = R_list_v_ori->head->next;
	R_node_v2 = R_list_v_appr->head->next;
	while( R_node_v2 != NULL)
	{
		//t
		if( R_node_v1->id1 != R_node_v2->id1)
		{
			fprintf( stderr, "id inconsistency [collect_position_error].\n");
			exit( 0);
		}
		//t
		
		time_cnt = 0;
		while( R_node_v1->id2 != R_node_v2->id2)
		{
			if( dist_opt == 1)
				position_error_v = calc_per_dist( R_node_v1->loc_v2, R_node_v2->loc_v1, R_node_v2->loc_v2, PERPENDICULAR_OPTION);
			else
			{
				time_cnt += R_node_v1->t_interval;
				
				position_error_v = calc_syn_Euclidean_dist( R_node_v1->loc_v2, R_node_v2->loc_v1, R_node_v2->loc_v2, time_cnt / R_node_v2->t_interval);
			}

			if( position_error_v > position_error_max)
				position_error_max = position_error_v;

			R_node_v1 = R_node_v1->next;
		}
		R_node_v1 = R_node_v1->next;

		R_node_v2 = R_node_v2->next;
	}
	
	*position_error = position_error_max;
}

/*
 *	Collect the position error of a simplified trajectory @R_list_v_appr.
 */
void collect_direction_error( R_list_t* R_list_v_ori, R_list_t* R_list_v_appr, float* direction_error)
{
	float direction_err_max, direction_err_v;
	R_node_t* R_node_v1, *R_node_v2;

	direction_err_max = 0;
	R_node_v1 = R_list_v_ori->head->next;
	R_node_v2 = R_list_v_appr->head->next;
	while( R_node_v2 != NULL)
	{
		//t
		if( R_node_v1->id1 != R_node_v2->id1)
		{
			fprintf( stderr, "R_list_v_appr inconsistency. [collect_direction_error]\n");
			exit( 0);
		}
		//t


		while( R_node_v1 != NULL)
		{
			//
			direction_err_v = calc_angle( R_node_v1->loc_v1,R_node_v1->loc_v2, R_node_v2->loc_v1, R_node_v2->loc_v2);

			if( direction_err_v > direction_err_max)
				direction_err_max = direction_err_v;

			if( R_node_v1->id2 == R_node_v2->id2)
				break;

			R_node_v1 = R_node_v1->next;
		}

		R_node_v1 = R_node_v1->next;
		R_node_v2 = R_node_v2->next;
	}

	*direction_error = direction_err_max;

	return;
}

//For testing only.
void print_error_matrix( R_list_t* R_list_v, int p_opt)
{
	int i;
	float error_v;
	R_node_t* R_node_v1, *R_node_v2;


	FILE* o_fp;

	if( p_opt == 1)
		o_fp = stdout;
	else
		o_fp = fopen( TEMP_FILE, "w");

	if( R_list_v == NULL)
		return;

	fprintf( o_fp, "\t");
	for( i=0; i<=R_list_v->size; i++)
		fprintf( o_fp, "%i\t", i+1);
	fprintf( o_fp, "\n");

	i = 1;
	R_node_v1 = R_list_v->head->next;
	while( R_node_v1 != NULL)
	{
		fprintf( o_fp, "%i\t", i++);

		R_node_v2 = R_list_v->head->next;
		while( R_node_v2 != R_node_v1)
		{
			fprintf( o_fp, "0\t");

			R_node_v2 = R_node_v2->next;
		}

		fprintf( o_fp, "0\t0\t");

		R_node_v2 = R_node_v1->next;
		while( R_node_v2 != NULL)
		{
			error_v = calc_error( R_node_v1, R_node_v2);

			fprintf( o_fp, "%.2f\t", error_v);

			R_node_v2 = R_node_v2->next;
		}

		fprintf( o_fp, "\n");

		R_node_v1 = R_node_v1->next;
	}
	fprintf( o_fp, "%i\t", i);
	for( i=0; i<=R_list_v->size; i++)
		fprintf( o_fp, "%i\t", 0);
	fprintf( o_fp, "\n");
}

/*
 *	Print a hop_list_t struct.
 */
void print_hop_list( hop_list_t* hop_list_v)
{
	hop_node_t* hop_node_v;

	if( hop_list_v == NULL)
	{
		printf( "Empty hop_list.\n");
		return;
	}

	hop_node_v = hop_list_v->head->next;
	while( hop_node_v != NULL)
	{
		printf( "%i  ", hop_node_v->pos_id);

		hop_node_v = hop_node_v->next;
	}

	printf( "\n");

	return;
}

/*
 *	Print a slope_r_list struct.
 */
void print_slope_r_list( slope_r_list_t* slope_r_list_v)
{
	slope_r_node_t* iter;

	if( slope_r_list_v == NULL)
	{
		printf( "Empty slope_r_list.\n");
		return;
	}

	iter = slope_r_list_v->head->next;
	while( iter != NULL)
	{
		printf( "[%f, %f]\t", iter->slope_r_v.s_sta, iter->slope_r_v.s_end);

		iter = iter->next;
	}

	printf( "\n");
	return;
}

//Empirical study.
/*
 *	The interface of all algorithms for DPTS.
 *
 *	@tra_list_t is the input trajectory.
 *	@eps is the error tolerance.
 *	@alg_opt indicates the simplification to be used.
 *		=0: the basic DP
 *		=1: the enhanced DP
 *		=2: the basic SP
 *		=3:	the SP with the practical enhancement only
 *		=4: the SP with the theorectical enhancement only
 *		=5: the SP with both the practical enhancement and the theoretical one
 *		=6: Intersect
 *		=7: Split
 *		=8: Merge
 *		=9: Greedy_v1
 *		=10: Greedy_v2
 *
 *	@Return the size of the simplified trajectory.
 */
int DPTS( tra_list_t* tra_list_v, float eps, int alg_opt)
{
	int simp_size, quota;
	float tmp1, tmp2;
	
	R_list_t* R_list_v, *R_list_v_appr, *R_list_v_appr_pos_per, *R_list_v_appr_pos_syn, *R_list_v_appr_pos_per_2, *R_list_v_appr_pos_syn_2;
	id_list_t* id_list_v, *id_list_v_pos_per, *id_list_v_pos_syn, *id_list_v_pos_per_2, *id_list_v_pos_syn_2;

	id_list_v = NULL;
	R_list_v = NULL;

	R_list_v = R_list_transform( tra_list_v);
	if( R_list_v == NULL)
		return tra_list_v->pos_n;

	/*t/
	print_error_matrix( R_list_v, 2);
	/*t*/

	if( alg_opt == 0)
	{
		printf( "the basic DP ...\n");
		simp_size = DP_Number( R_list_v, eps, 0);

		simp_size += 1;
	}
	else if( alg_opt == 1)
	{
		printf( "the enhanced DP ...\n");
		simp_size = DP_Number( R_list_v, eps, 1);

		simp_size += 1;	//
	}
	else if( alg_opt == 2)
	{
		printf( "SP-basic ...\n");
		simp_size = SP_Method_B( R_list_v, eps);

		simp_size += 1;	//SP_Method_7 returns the length of the SP.
	}
	else if( alg_opt == 3)
	{
		printf( "SP-practical ...\n");
		id_list_v = SP_Method_prac( R_list_v, eps);	
	}
	else if( alg_opt == 4)
	{
		printf( "SP-theoretical ...\n");
		simp_size = SP_Method_theo( R_list_v, eps);

		simp_size += 1;	//SP_Method_B returns the length of the SP.
	}
	else if( alg_opt == 5)
	{
		printf( "SP-both ...\n");
		id_list_v = SP_Method_prac_and_theo( R_list_v, eps);
	}
	else if( alg_opt == 6)
	{
		printf( "Intersect ...\n");
		id_list_v = Intersect( R_list_v, eps);
	}
	else if( alg_opt == 7)
	{
		printf( "Split ...\n");
		simp_size = Split( R_list_v, eps);
	}
	else if( alg_opt == 8)
	{
		printf( "Merge ...\n");
		id_list_v = Merge( R_list_v, eps);
	}
	else if( alg_opt == 9)
	{
		printf( "Greedy_v1 ...\n");
		id_list_v = Greedy_v1( R_list_v, eps);
	}
	else //if( alg_opt == 10)
	{
		printf( "Greedy_v2 ...\n");
		id_list_v = Greedy_v2( R_list_v, eps);
	}
	
	if( id_list_v != NULL)
	{
		/*t/
		id_list_print( id_list_v);
		printf( "\n");
		/*t*/

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
	
	/*t*/
	printf( "Original size: %i\n", R_list_v->size + 1);
	printf( "Simplified size: %i\n", simp_size);
	printf( "Compression ratio: %f\n\n", ( float)simp_size / ( R_list_v->size + 1));
	/*t*/

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


	id_list_release( id_list_v);
	R_list_release( R_list_v);

	return simp_size;
}

/*
 *	Empirical study for DPTS.
 */
void emp_DPTS( )
{
	int simp_size;

	emp_config_t* emp_config_v;
	tra_list_t* tra_list_v;

	FILE* s_fp;

	//Pre-set the statistics collector emp_stat_v.
	memset( &emp_stat_v, 0, sizeof( emp_stat_t));
	emp_stat_v.C.min = FLT_MAX;

	//Read the config.
	emp_config_v = emp_read_config( );

	//Read the data.
	//printf( "Reading data ...\n");
	if( emp_config_v->dataset_tag == 1)
		tra_list_v = emp_read_data_v1( emp_config_v);
	else
		tra_list_v = emp_read_data_v2( emp_config_v);

#ifndef WIN32
	rusage before_simp_t;
	get_cur_time( &before_simp_t);
#endif
	
	//Perform the simplification process.
	printf( "Simplifying ...\n");
	simp_size = DPTS( tra_list_v, emp_config_v->eps, emp_config_v->alg_opt);
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
	emp_stat_v.cmp_ratio = ( float)simp_size / emp_config_v->pos_n;

	//percent.
	emp_stat_v.percent = emp_stat_v.eps_n / ( emp_config_v->pos_n * ( emp_config_v->pos_n - 1) / 2);

	//C.aver.
	if( emp_stat_v.cap_n != 0)
		emp_stat_v.C.aver = emp_stat_v.C.sum / emp_stat_v.cap_n;

	//Print the results & statistics.
	if( ( s_fp = fopen( STATISTIC_FILE, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open the stat.txt\n");
		exit( 0);
	}

	//Simplification results.
	fprintf( s_fp, "%i\n%i\n%f\n\n", tra_list_v->pos_n, simp_size, emp_stat_v.cmp_ratio);

	//Statistics.
	fprintf( s_fp, "%f\n%f\n\n", emp_stat_v.r_time, emp_stat_v.memory_max / (1024 * 1024));

	fprintf( s_fp, "%f\n\n", emp_stat_v.percent);

	if( emp_stat_v.C.min == FLT_MAX)
		emp_stat_v.C.min = 1;
	fprintf( s_fp, "%f\n%f\n%f\n\n", emp_stat_v.C.min, emp_stat_v.C.max, emp_stat_v.C.aver);

	
	emp_stat_v.length_err_bound = cos( emp_config_v->eps);
	emp_stat_v.speed_err_bound = cos( emp_config_v->eps);

	fprintf( s_fp, "%f\n%f\n\n%f\n%f\n\n", emp_stat_v.length_err, emp_stat_v.length_err_bound, 
		emp_stat_v.speed_err, emp_stat_v.speed_err_bound);
	
	fprintf( s_fp, "%f\n%f\n%f\n%f\n\n%f\n%f\n%f\n%f\n\n", 
		emp_stat_v.pos_err_per, emp_stat_v.pos_err_per_heur, emp_stat_v.pos_err_per_opti, emp_stat_v.pos_err_per_bound,
		emp_stat_v.pos_err_syn, emp_stat_v.pos_err_syn_heur, emp_stat_v.pos_err_syn_opti, emp_stat_v.pos_err_syn_bound);

	fprintf( s_fp, "%f\n%f\n%f\n%f\n%f\n\n", emp_stat_v.dir_err_dpts, 
									emp_stat_v.dir_err_per_heur, emp_stat_v.dir_err_per_opti, 
									emp_stat_v.dir_err_syn_heur, emp_stat_v.dir_err_syn_opti);

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
	tra_list_release( tra_list_v);
	free( emp_config_v);

	printf( "Running time: %f\n", emp_stat_v.r_time);
	printf( "Number of checkings: %f\n", emp_stat_v.eps_n);
	printf( "Memory balance: %f\n", emp_stat_v.memory_v);
}




/*
 *	Read the raw trajectory.
 */
tra_list_t* read_trajectory_v1( char* f_name)
{
	int i;
	float pre_x, pre_y;
	char str_tmp[ MAX_LINE_LENG];
	char* tok;

	tra_list_t* raw_data;
	triplet_t tri_v;
	FILE* i_fp;

	if( ( i_fp = fopen( f_name, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open %s.\n", f_name);
		exit( 0);
	}		

	raw_data = tra_list_alloc( );
	
	//Filter the useless lines;
	for( i=0; i<NUM_LINES_IGNORE; i++)
		fgets( str_tmp, MAX_LINE_LENG, i_fp);

	//Read the file.
	pre_x = FLT_MAX;
	pre_y = FLT_MIN;
	while( fgets( str_tmp, MAX_LINE_LENG, i_fp))
	{
		//Parse the line to get the x and y coordinates.
		tok = strtok( str_tmp, ",");
		tri_v.loc_v.x = atof( tok);

		tok = strtok( NULL, ",");
		tri_v.loc_v.y = atof( tok);

		/*
		if( fabs( tri_v.loc_v.x - pre_x) <= emp_config_v->precision_thr &&
			fabs( tri_v.loc_v.y - pre_y) <= emp_config_v->precision_thr)
			continue;
		*/

		pre_x = tri_v.loc_v.x;
		pre_y = tri_v.loc_v.y;

		//
		//tri_v.t = 0;

		/*t/
		printf( "%f\t%f\n", tri_v.loc_v.x, tri_v.loc_v.y);
		/*t*/

		append_tra_list_entry( raw_data, tri_v);
	}

	//Release the resources.
	fclose( i_fp);
	
	return raw_data;
}

/*
 *
 */
tra_list_t* read_trajectory_v2( FILE* i_fp)
{	
	int i, id, pos_n;
	float timer;
	triplet_t tri_v;
	tra_list_t* raw_data;

	//Use a timer.
	timer = 0;

	raw_data = tra_list_alloc( );

	fscanf( i_fp, "%i%i", &id, &pos_n);

	for( i=0; i<pos_n; i++)
	{
		memset( &tri_v, 0, sizeof( tri_v));
		fscanf( i_fp, "%f%f", &tri_v.loc_v.x, &tri_v.loc_v.y);
		//tri_v.t = 0;
		tri_v.sec = timer ++;

		append_tra_list_entry( raw_data, tri_v);
	}

	return raw_data;
}

/*
 *	For reading T-Drive.
 */
tra_list_t* read_trajectory_v3( char* f_name)
{
	float pre_x, pre_y;
	char str_tmp[ MAX_LINE_LENG];
	char* tok;

	tra_list_t* raw_data;
	triplet_t tri_v;
	FILE* i_fp;

	if( ( i_fp = fopen( f_name, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open %s.\n", f_name);
		exit( 0);
	}		

	raw_data = tra_list_alloc( );
	
	//Filter the useless lines;
	//for( i=0; i<NUM_LINES_IGNORE; i++)
	//	fgets( str_tmp, MAX_LINE_LENG, i_fp);


	//10039,2008-02-02 13:48:49,116.51704,40.01651

	//Read the file.
	pre_x = FLT_MAX;
	pre_y = FLT_MIN;
	while( fgets( str_tmp, MAX_LINE_LENG, i_fp))
	{
		//Parse the line to get the x and y coordinates.
		tok = strtok( str_tmp, ",");
		tok = strtok( NULL, ",");

		tok = strtok( NULL, ",");
		tri_v.loc_v.x = atof( tok);

		tok = strtok( NULL, ",");
		tri_v.loc_v.y = atof( tok);

		/*
		if( fabs( tri_v.loc_v.x - pre_x) <= emp_config_v->precision_thr &&
			fabs( tri_v.loc_v.y - pre_y) <= emp_config_v->precision_thr)
			continue;
		*/

		pre_x = tri_v.loc_v.x;
		pre_y = tri_v.loc_v.y;

		//
		//tri_v.t = 0;

		/*t/
		printf( "%f\t%f\n", tri_v.loc_v.x, tri_v.loc_v.y);
		/*t*/

		append_tra_list_entry( raw_data, tri_v);
	}

	//Release the resources.
	fclose( i_fp);
	
	return raw_data;
}


/*
 *	Retrieve the simplifid trajectory.
 */
tra_list_t* retrieve_trajectory( tra_list_t* raw_data, id_list_t* id_list_v)
{
	id_node_t* id_node_v;
	tra_list_t* simp_data;
	tra_node_t* tra_node_v;

	if( id_list_v->pos_id_n == 0)
		return tra_list_alloc( );

	simp_data = tra_list_alloc( );
	tra_node_v = raw_data->head->next;

	/*t/
	printf( "Tra_list size: %i\n", raw_data->pos_n);
	printf( "id_list size: %i\n", id_list_v->pos_id_n);
	/*t*/
	
	id_node_v = id_list_v->head->next;
	while( id_node_v != NULL)
	{
		/*t/
		printf( "----%i\n", id_node_v->pos_id);
		/*t*/

		while( tra_node_v->pos_id != id_node_v->pos_id)
		{
			tra_node_v = tra_node_v->next;

			/*t/
			printf( "%i\n", tra_node_v->pos_id);
			/*t*/
		}
		
		append_tra_list_entry( simp_data, tra_node_v->tri_v);

		id_node_v = id_node_v->next;
	}

	return simp_data;
}

/*
 *	Perform a direction-preserving simplification.
 */
tra_list_t* dir_simp_trajectory( tra_list_t* raw_data, float eps)
{
	id_list_t* id_list_v;
	R_list_t* R_list_v;
	tra_list_t* simp_data;

	R_list_v = R_list_transform( raw_data);

	id_list_v = SP_Method_prac_and_theo( R_list_v, eps);
	//id_list_v = SP_Method_prac( R_list_v, eps);

	simp_data = retrieve_trajectory( raw_data, id_list_v);

	/*t*/
	//printf( "here?\n");
	/*t*/
	
	R_list_release( R_list_v);
	id_list_release( id_list_v);
	
	return simp_data;
}

/*
 *	Allocate a part_list_t struct.
 */
part_list_t* part_list_alloc( )
{
	part_list_t* part_list_v;

	part_list_v = ( part_list_t*)malloc( sizeof( part_list_t));
	memset( part_list_v, 0, sizeof( part_list_t));

	part_list_v->head = ( part_node_t*)malloc( sizeof( part_node_t));
	memset( part_list_v->head, 0, sizeof( part_node_t));

	part_list_v->rear = part_list_v->head;

	/*s*/
	emp_stat_v.memory_v += sizeof(part_list_t) + sizeof( part_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/

	return part_list_v;
}

/*
 *	Release a part_list_t struct.
 */
void part_list_release( part_list_t* part_list_v)
{
	part_node_t* part_node_v1, *part_node_v2;

	part_node_v1 = part_list_v->head;
	while( part_node_v1 != NULL)
	{
		part_node_v2 = part_node_v1->next;
		
		free( part_node_v1);

		part_node_v1 = part_node_v2;
	}

	/*s*/
	emp_stat_v.memory_v -= ( part_list_v->size + 1) * sizeof( part_node_t) + sizeof( part_list_t);
	/*s*/

	free( part_list_v);
	return;
}

/*
 *	Calculate the perpendicular distance from a point @loc_v1 to the line segment with its end points of seg_end1 and seg_end2.
 *
 *	@opt
 *		= 1: the perpendicular distance (may beyond the segment).
 *		= 2: the shortest distance (must within the segment).
 *
 */
double calc_per_dist( loc_t* loc_v1, loc_t* seg_end1, loc_t* seg_end2, int opt)
{
	loc_t p;
	double u;//, dist_v1, dist_v2;

//	if( opt == 2 && !( min( seg_end1->x, seg_end2->x) <= loc_v1->x && loc_v1->x <= max( seg_end1->x, seg_end2->x)
//					|| min( seg_end1->y, seg_end2->y) <= loc_v1->y && loc_v1->y <= max( seg_end1->y, seg_end2->y)))
//	{
//		dist_v1 = calc_Euclidean_dist( loc_v1, seg_end1);
//		dist_v2 = calc_Euclidean_dist( loc_v1, seg_end2);
//
//		return min( dist_v1, dist_v2);
//	}
//	else
//	{
		u = calc_dot_product( seg_end1, loc_v1, seg_end1, seg_end2) / 
			pow( calc_Euclidean_dist( seg_end1, seg_end2), 2);
		
		//p is the projection of loc_v1 on the line that passes through the segment.
		p.x = seg_end1->x + u * ( seg_end2->x - seg_end1->x);
		p.y = seg_end1->y + u * ( seg_end2->y - seg_end1->y);

		if( opt == 2 && ( u < 0 || u > 1))
		{
			if( u < 0)
				return calc_Euclidean_dist( loc_v1, seg_end1);
			else
				return calc_Euclidean_dist( loc_v1, seg_end2);
		}
		else
			return calc_Euclidean_dist( loc_v1, &p);
//	}
}

/*
 *	Compute the "synchronous Euclidean dist" of a point @loc_v1.
 */
double calc_syn_Euclidean_dist( loc_t* loc_v1, loc_t* seg_end1, loc_t* seg_end2, float time_portion)
{
	loc_t p;

	p.x = seg_end1->x + time_portion * ( seg_end2->x - seg_end1->x);
	p.y = seg_end1->y + time_portion * ( seg_end2->y - seg_end1->y);
	
	return calc_Euclidean_dist( loc_v1, &p);
}

/*
 *	Calculate the largest distance corresponding to a group of positions/segs.
 *
 *	@dist_opt 
 *		= 1: the perpendicular distance.
 *		= 2: the synchronous Euclidean distance.
 */
void ini_part_node_dist( part_node_t* part_node_v, int dist_opt)
{	
	float time_cnt, time_interval;
	double dist, max_dist;
	
	R_node_t* iter;

	max_dist = 0;
	//part_node_v->pos_id_split = part_node_v->end->id1;
	part_node_v->split_loc = part_node_v->end;

	//
	time_interval = 0;
	iter = part_node_v->sta;
	while( iter != NULL)
	{
		time_interval += iter->t_interval;

		if( iter == part_node_v->end)
			break;

		iter = iter->next;
	}	

	//
	time_cnt = 0;
	iter = part_node_v->sta;
	while( iter != part_node_v->end)
	{
		if( dist_opt == 1)	//Perpendicular distance.
			dist = calc_per_dist( iter->loc_v2, part_node_v->sta->loc_v1, part_node_v->end->loc_v2, PERPENDICULAR_OPTION);
		else				//Synchronous Euclidean distance.
		{
			//Compute the portion.
			time_cnt += iter->t_interval;

			dist = calc_syn_Euclidean_dist( iter->loc_v2, part_node_v->sta->loc_v1, part_node_v->end->loc_v2, time_cnt / time_interval);
		}
	
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
 *	A Dogoulas-Peucker-like algorithm.
 */
id_list_t* position_TS( R_list_t* R_list_v, int quota, int dist_opt)
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

	ini_part_node_dist( part_node_v, dist_opt);
	part_list_v->head->next = part_node_v;
	part_list_v->rear = part_node_v;
	part_list_v->size ++;	

	/*s*/
	emp_stat_v.memory_v += sizeof( part_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/
	
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
	while( part_list_v->size < quota)
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

		/*s*/
		emp_stat_v.memory_v += sizeof( part_node_t);
		if( emp_stat_v.memory_v > emp_stat_v.memory_max)
			emp_stat_v.memory_max = emp_stat_v.memory_v;
		/*s*/

		//Re-set the attributes.
		ini_part_node_dist( part_node_v1, dist_opt);
		ini_part_node_dist( part_node_v2, dist_opt);

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
 *	Compute the approximation error using the position-based error measurements.
 */
DIST_TYPE calc_error_dist( R_node_t* R_node_v1, R_node_t* R_node_v2, int dist_opt)
{
	float time_cnt, time_interval;
	DIST_TYPE error_v, error_max;
	
	R_node_t* R_node_v;


	if( dist_opt == 2)
	{
		//Collect the time_interval information.
		time_interval = 0;

		R_node_v = R_node_v1;
		while( R_node_v != NULL)
		{
			time_interval += R_node_v->t_interval;

			if( R_node_v == R_node_v2)
				break;

			R_node_v = R_node_v->next;
		}
	}
	
	time_cnt = 0;
	error_max = 0;
	R_node_v = R_node_v1;
	while( R_node_v != NULL)
	{
		if( R_node_v == R_node_v2)
			break;

		//Compute the error incurred by R_node_v->loc_v2;
		if( dist_opt == 1)
			error_v = calc_per_dist( R_node_v->loc_v2, R_node_v1->loc_v1, R_node_v2->loc_v2, PERPENDICULAR_OPTION);
		else
		{
			time_cnt += R_node_v->t_interval;
			error_v = calc_syn_Euclidean_dist( R_node_v->loc_v2, R_node_v1->loc_v1, R_node_v2->loc_v2, time_cnt / time_interval);
		}

		if( error_v > error_max)
			error_max = error_v;

		R_node_v = R_node_v->next;
	}

	return error_max;
}

/*
 *	The optimal algorithm of the min-error problem (using position-based measurements).
 *
 *	Maintain a table of error.
 */
DIST_TYPE position_TS_DP( R_list_t* R_list_v, int quota, int dist_opt)
{
	int i, j, k, pos_n, cnt, L;
	float error_min, error_v, error_v1, error_v2, res;
	float** error_matrix, **error_table;

	R_node_t** R_node_v_array, *iter;

	L = quota;
	pos_n = R_list_v->size + 1;

	error_matrix = ( float**)malloc( sizeof( float*) * pos_n);
	memset( error_matrix, 0, sizeof( float*) * pos_n);
	error_table = ( float**)malloc( sizeof( float*) * pos_n);
	memset( error_table, 0, sizeof( float*) * pos_n);

	for( i=0; i<pos_n; i++)
	{
		error_matrix[ i] = ( float*)malloc( sizeof( float) * ( L + 1));
		memset( error_matrix[ i], 0, sizeof( float) * ( L + 1));

		error_table[ i] = ( float*)malloc( sizeof( float) * pos_n);
		memset( error_table[ i], 0, sizeof( float) * pos_n);
	}


	R_node_v_array = ( R_node_t**)malloc( sizeof( R_node_t*) * pos_n);
	memset( R_node_v_array, 0, sizeof( R_node_t*) * pos_n);
	
	cnt = 1;
	iter = R_list_v->head->next;
	while( iter != NULL)
	{
		R_node_v_array[ cnt++] = iter;

		iter = iter->next;
	}

	//Maintain the error table.
	for( i=1; i<pos_n; i++)
	{
		for( j=i+1; j<pos_n; j++)
		{
			error_table[ i][ j] = calc_error_dist( R_node_v_array[ i], R_node_v_array[ j], dist_opt);
		}

		//t
		//printf( "%i\n", i);
	}


	//t
	if( cnt != pos_n)
	{
		fprintf( stderr, "R_list_v inconsistency. [position_TS_DP]\n");
		exit( 0);
	}
	//t

	//Initialize the errors for the boundary cases.
	//Boundary Case 1: error_matrix[ pos_n - 1][ * ].
	for( j=1; j<=L; j++)
		error_matrix[ pos_n - 1][ j] = 0;

	//Boundary Case 2: the quota is equal to 1.
	for( i=1; i<pos_n; i++)
		error_matrix[ i][ 1] = error_table[ i][ pos_n-1];//calc_error_dist( R_node_v_array[ i], R_node_v_array[ pos_n-1], dist_opt);

	//Deal with the normal cases.
	for( j=2; j<=L; j++)
	{
		for( i=pos_n-2; i>=1; i--)
		{
			//Compute error_matrix[ i][ j].
			//t
		//	printf( "%i\t%i\n", i, j);
			//t

			error_min = FLT_MAX;
			for( k=i+1; k<pos_n; k++)
			{
				error_v1 = error_table[ i][ k-1];//calc_error_dist( R_node_v_array[ i], R_node_v_array[ k-1], dist_opt);

				error_v2 = error_matrix[ k][ j-1];

				error_v = max( error_v1, error_v2);

				if( error_v < error_min)
					error_min = error_v;
			}

			error_matrix[ i][ j] = error_min;
		}
	}//

	res = error_matrix[ 1][ L];

	//Release the resources.
	free( R_node_v_array);
	for( i=0; i<pos_n; i++)
	{
		free( error_matrix[ i]);
		free( error_table[ i]);
	}
	
	free( error_matrix);
	free( error_table);

	return res;
}


/*
 *	This is identical to SP_Method_prac except for the part of checking errors.
 */
id_list_t* SP_Method_prac_dist( R_list_t* R_list_v, float eps, int dist_opt)
{
	int seg_n, pos_n, pos_id1, pos_id2, error_i, i, cur_id, cnt;
	int *pre;
	
	hop_list_t* H, *U, *hop_list_v, *pre_hop_list_v;
	hop_node_t* hop_node_v1, *hop_node_v2, *hop_node_v3;
	
	R_node_t* seg_before, *seg_after;
	
	id_list_t* id_list_v;
	
	
	seg_n = R_list_v->size;
	pos_n = seg_n + 1;
	
	pre = ( int*)malloc( pos_n * sizeof( int));
	memset( pre, 0, pos_n * sizeof( int));
	
	/*s*/
	emp_stat_v.memory_v += pos_n * sizeof( int);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	/*s*/
	
	//Initialize H_0
	H = hop_list_alloc( );	
	seg_before = NULL;
	seg_after = R_list_v->head->next;
	append_hop_list_entry( H, 0, seg_before, seg_after);
	
	pre[ 0] = -1;
	
	
	//Initialize U.
	U = hop_list_alloc( );
	for( i=1; i<pos_n; i++)
	{
		seg_before = seg_after;
		seg_after = seg_before->next;
		//append_hop_list_entry( U, i, seg_before, seg_after);
		
		ahead_hop_list_entry( U, i, seg_before, seg_after);
	}
	
	
	//Compute H_h's for h=1, 2, ...
	pre_hop_list_v = H;
	cnt = 0;
	while( true)
	{
		/*t/
		printf( "%i-length set: \n", cnt ++);
		//print_hop_list( pre_hop_list_v);
		/*t*/
		
		hop_list_v = hop_list_alloc( );
		pre_hop_list_v->next = hop_list_v;
		
		hop_node_v1 = pre_hop_list_v->head->next;
		while( hop_node_v1 != NULL)
		{			
			pos_id1 = hop_node_v1->pos_id;

			/*t/
			printf( "U: ");
			print_hop_list( U);
			/*t*/
			
			
			//hop_node_v3 = U->head;
			hop_node_v2 = U->head->next;
			while( hop_node_v2 != NULL)
			{
				if( hop_node_v2->pos_id <= hop_node_v1->pos_id)
					break;			//
				
				pos_id2 = hop_node_v2->pos_id;
				
				//Compute the error.
				
				//Option 1: using the straightforward method (O(n)).
				//error_i = check_error( hop_node_v1->seg_after, hop_node_v2->seg_before, eps);
				if( calc_error_dist( hop_node_v1->seg_after, hop_node_v2->seg_before, dist_opt) <= eps)
					error_i = 1;
				else
					error_i = 0;
				
				/*s*/
				emp_stat_v.eps_n ++;
				/*s*/
				
				
				if( error_i)
				{
					if( pos_id2 == pos_n - 1)
					{
						//The shortest path has been founded.
						//Check out the shortest path.
						pre[ pos_id2] = pos_id1;
						
						id_list_v = id_list_alloc( );
						
						cur_id = pos_id2;
						//append_id_list_entry( id_list_v, cur_id);
						ahead_id_list_entry( id_list_v, cur_id);
						while( pre[ cur_id] != -1)
						{
							/*t/
							if( pre[ cur_id] < 0)
								printf( "");
							/*t*/
							
							cur_id = pre[ cur_id];
							
							ahead_id_list_entry( id_list_v, cur_id);
						}
						
						//Release the hop_list_t struct.
						hop_list_release( H);
						hop_list_release( U);
						
						
						free( pre);
						
						/*s*/
						emp_stat_v.memory_v -= pos_n * sizeof( int);
						/*s*/
						
						return id_list_v;
					}
					
					//Remove the position with id2 from U;
					hop_node_v3 = hop_node_v2->pre;
					hop_node_v3->next = hop_node_v2->next;
					if( hop_node_v2->next != NULL)
					{
						hop_node_v2->next->pre = hop_node_v3;
					}
					
					if( U->rear == hop_node_v2)
						U->rear = hop_node_v3;
					
					//Add the position with id2 into H_h;
					hop_node_v2->next = NULL;
					hop_node_v2->pre = hop_list_v->rear;
					hop_list_v->rear->next = hop_node_v2;
					hop_list_v->rear = hop_node_v2;
					
					pre[ pos_id2] = pos_id1;				
					
					hop_node_v2 = hop_node_v3;
				}//if( error_i)
				
				hop_node_v2 = hop_node_v2->next;							
			}//while( hop_node_v2)
			
			
			hop_node_v1 = hop_node_v1->next;
			
		}//while( hop_node_v1)
		
		pre_hop_list_v = hop_list_v;		
	}//while(true)
	
	/*t*/
	fprintf( stderr, "Bug in [SP_Method_sub2]...\n");
	exit( 0);
	/*t*/
}

/*
 *	Perform a position-preserving simplification.
 *
 */
tra_list_t* dist_simp_trajectory( tra_list_t* raw_data, int quota, int dist_opt)
{
	id_list_t* id_list_v;
	R_list_t* R_list_v;
	tra_list_t* simp_data;

	
	R_list_v = R_list_transform( raw_data);

	id_list_v = position_TS( R_list_v, quota, dist_opt);

	simp_data = retrieve_trajectory( raw_data, id_list_v);

	
	R_list_release( R_list_v);
	id_list_release( id_list_v);
	
	return simp_data;
}

/*
 *	Print a tra_list to a file.
 */
void print_tra_list( tra_list_t* tra_list_v, FILE* o_fp)
{
	tra_node_t* iter;

	if( tra_list_v == NULL)
		return;

	iter = tra_list_v->head->next;
	while( iter != NULL)
	{
		fprintf( o_fp, "%0.1f  %0.1f  ", iter->tri_v.loc_v.x, iter->tri_v.loc_v.y);
		
		iter = iter->next;
	}

	return;
}

#ifndef WIN32
//For the clustering application.
/*
 *	Prepare the datasets for clustering use.
 *
 *		Prepare 3 types of data.
 *			1: Use the raw trajectory data.
 *			2: Use the direction-preserving simplified trajectory data.
 *			3: Use the position-preserving simplified trajectory data.
 *
 *	This version works in Linux and traverse a specific folder.
 *	
 */
void prepare_cluster_data_v1(  )
{
	int ins_n, file_cnt, tra_cnt, quota;
	float eps;//, ori_pos_cnt, simp_pos_cnt;;
	char dir_path[ MAX_PATHNAME_LENG];
	char in_f_name[ MAX_FILENAME_LENG];
	char out_f_name[ MAX_FILENAME_LENG];
	char f_name_tmp[ MAX_FILENAME_LENG];
	char f_name_tmp1[ MAX_FILENAME_LENG];
	char f_name_tmp2[ MAX_FILENAME_LENG];
	char f_name_tmp3[ MAX_FILENAME_LENG];
	char f_name_tmp4[ MAX_FILENAME_LENG];
	char f_name_tmp5[ MAX_FILENAME_LENG];
	char str_tmp[ MAX_STR_LENG];
	//char** file_names;

	tra_list_t* raw_data, *dir_simp_data, *dist_simp_data_per, *dist_simp_data_syn;
	
	FILE* c_fp, *i_fp, *o_fp1, *o_fp2, *o_fp3, *o_fp4;
	
	DIR* dir_p;
	struct dirent* dir_r;
	unsigned char is_file =0x8;

	if( ( c_fp = fopen( CLUSTER_DATA_CONFIG, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open cluster_data_config.txt\n");
		exit( 0);
	}

	//Handle the instances
	ins_n = 1;
	while( fscanf( c_fp, "%s%s%f", dir_path, out_f_name, &eps))
	{		
		//Handle a specific case.
		printf( "Handling Case #%i ...\n", ins_n ++);

		//Prepare the output file.
		strcpy( f_name_tmp, CLUSTER_DATA_LOC);
		strcat( f_name_tmp, out_f_name);
		sprintf( str_tmp, "_%0.3f", eps);
		strcat( f_name_tmp, str_tmp);

		//raw data.
		strcpy( f_name_tmp1, f_name_tmp);
		strcat( f_name_tmp1, "_raw");

		//direction-preserving simplified data.
		strcpy( f_name_tmp2, f_name_tmp);
		strcat( f_name_tmp2, "_dir");
		
		//distance-preserving simplified data (perpendicular).
		strcpy( f_name_tmp3, f_name_tmp);
		strcat( f_name_tmp3, "_pos_per");

		//distance-preserving simplified data (perpendicular).
		strcpy( f_name_tmp5, f_name_tmp);
		strcat( f_name_tmp5, "_pos_syn");

		if( ( o_fp1 = fopen( f_name_tmp1, "w")) == NULL ||
			( o_fp2 = fopen( f_name_tmp2, "w")) == NULL ||
			( o_fp3 = fopen( f_name_tmp3, "w")) == NULL ||
			( o_fp4 = fopen( f_name_tmp5, "w")) == NULL)
		{
			fprintf( stderr, "Cannot open the %s file.\n", f_name_tmp);
			exit( 0);
		}

		//Count the trajectories in the directory.
		if( ( dir_p = opendir( dir_path)) == NULL)
		{
			fprintf( stderr, "Error occurs when opening %s.\n", dir_path);
			exit( 0);
		}

		file_cnt = 0;
		while( ( dir_r = readdir( dir_p)) != NULL)
		{
			if( dir_r->d_type == is_file)
				file_cnt ++;
		}

		closedir( dir_p);
		

		/*t*/
		printf( "# of files: %i\n", file_cnt);
		/*t*/

		//Read the directory.
		if( ( dir_p = opendir( dir_path)) == NULL)
		{
			fprintf( stderr, "Error occurs when opening %s.\n", dir_path);
			exit( 0);
		}
		
		fprintf( o_fp1, "%i\n%i\n", 2, file_cnt);
		fprintf( o_fp2, "%i\n%i\n", 2, file_cnt);
		fprintf( o_fp3, "%i\n%i\n", 2, file_cnt);
		fprintf( o_fp4, "%i\n%i\n", 2, file_cnt);

		tra_cnt = -1;
		while( ( dir_r = readdir( dir_p)) != NULL)
		{
			//
			if( dir_r->d_type == is_file)
			{
				//It corresponds to a file.
				//dir_r->d_name is the file name.
				
				tra_cnt ++;

				//Handle the file dir_v->d_name.
				//Copy the raw data.
				{
					//Input: file name.
					//Output: a tra_list_t struct.
					strcpy( f_name_tmp4, dir_path);
					strcat( f_name_tmp4, dir_r->d_name);

					/*t*/
					printf( "Reading file %s ...\n", f_name_tmp4);
					/*t*/

					raw_data = read_trajectory_v1( f_name_tmp4);

					/*t/
					printf( "Finished reading!\n");
					/*t*/

					//Print the tra_list_t struct.
					fprintf( o_fp1, "%i  %i ", tra_cnt, raw_data->pos_n - 1);
					print_tra_list( raw_data, o_fp1);
					fprintf( o_fp1, "\n");

					//tra_list_release( tra_list_v);
				}
				
				//Perform the direction-preserving simplification.
				{
					//Input: raw data, eps.
					//Ouput: a tra_list_t struct.
					/*t*/
					printf( "DPTS ...\n");
					/*t*/

					dir_simp_data = dir_simp_trajectory( raw_data, eps);

					/*t/
					printf( "Finished!\n");
					/*t*/
					
					//Print the tra_list_t struct.
					fprintf( o_fp2, "%i  %i ", tra_cnt, dir_simp_data->pos_n - 1);
					print_tra_list( dir_simp_data, o_fp2);
					fprintf( o_fp2, "\n");
				}
				
				//Perfrom the position-preserving simplification (perpendicular distance).
				{
					//Input: raw data, the quota.
					//Output: a tra_list_ struct.
					//Algorithm Dougolas-Peucker.

					/*t*/
					printf( "Position-preserving TS (perpendicular distance) ...\n");
					/*t*/

					quota = dir_simp_data->pos_n - 1;
					dist_simp_data_per = dist_simp_trajectory( raw_data, quota, 1);
					
					/*t/
					printf( "Finished!\n");
					/*t*/

					//Print the tra_list_t struct
					fprintf( o_fp3, "%i  %i ", tra_cnt, dist_simp_data_per->pos_n - 1);
					print_tra_list( dist_simp_data_per, o_fp3);
					fprintf( o_fp3, "\n");
				}		

				//Perfrom the position-preserving simplification (synchronous Euclidean distance).
				{
					//Input: raw data, the quota.
					//Output: a tra_list_ struct.
					//Algorithm Dougolas-Peucker.

					/*t*/
					printf( "Position-preserving TS (synchronous Euclidean distance) ...\n");
					/*t*/

					quota = dir_simp_data->pos_n - 1;
					dist_simp_data_syn = dist_simp_trajectory( raw_data, quota, 2);
					
					/*t/
					printf( "Finished!\n");
					/*t*/

					//Print the tra_list_t struct
					fprintf( o_fp4, "%i  %i ", tra_cnt, dist_simp_data_syn->pos_n - 1);
					print_tra_list( dist_simp_data_syn, o_fp4);
					fprintf( o_fp4, "\n");
				}		

				//Release the resources.
				tra_list_release( raw_data);
				tra_list_release( dir_simp_data);
				tra_list_release( dist_simp_data_per);
				tra_list_release( dist_simp_data_syn);
				
			}//if( file)
		}//while( readdir)


		//Release the resources.
		closedir( dir_p);	
		
		fclose( o_fp1);
		fclose( o_fp2);
		fclose( o_fp3);
		fclose( o_fp4);

	}//while( fscanf)


	//Release the resources.
	fclose( c_fp);
}
#endif

/*
 *	This is the second version of prepare_cluster_data.
 *	Specific files (instead of a folder) should be specified.
 */
void prepare_cluster_data_v2( )
{
	int i, ins_n, tra_n, dim, quota;
	float eps;
	char raw_f_name[ MAX_FILENAME_LENG];
	char dir_f_name[ MAX_FILENAME_LENG];
	char dist_f_name_per[ MAX_FILENAME_LENG];
	char dist_f_name_syn[ MAX_FILENAME_LENG];

	tra_list_t* raw_data, *dir_simp_data, *dist_simp_data_per, *dist_simp_data_syn;

	FILE* c_fp, *i_fp, *o_fp1, *o_fp2, *o_fp3;

	if( ( c_fp = fopen( CLUSTER_DATA_CONFIG_V2, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open %s\n", CLUSTER_DATA_CONFIG_V2);
		exit( 0);
	}
	
	//Handling all the cases.
	ins_n = 1;
	while( fscanf( c_fp, "%s%f", raw_f_name, &eps) != EOF)
	{
		printf( "Handling Case #%i ...\n", ins_n ++);

		//
		sprintf( dir_f_name, "%s_dir_%0.3f", raw_f_name, eps);
		sprintf( dist_f_name_per, "%s_dist_per_%0.3f", raw_f_name, eps);
		sprintf( dist_f_name_syn, "%s_dist_syn_%0.3f", raw_f_name, eps);
		
		if( ( i_fp = fopen( raw_f_name, "r")) == NULL ||
			( o_fp1 = fopen( dir_f_name, "w")) == NULL ||
			( o_fp2 = fopen( dist_f_name_per, "w")) == NULL ||
			( o_fp3 = fopen( dist_f_name_syn, "w")) == NULL)
		{
			fprintf( stderr, "Cannot open the files. [prepare_cluster_data_v2]\n");
			exit( 0);
		}


		//
		fscanf( i_fp, "%i%i", &dim, &tra_n);
		fprintf( o_fp1, "%i\n%i\n", dim, tra_n);
		fprintf( o_fp2, "%i\n%i\n", dim, tra_n);
		fprintf( o_fp3, "%i\n%i\n", dim, tra_n);

		for( i=0; i<tra_n; i++)
		{
			//Input: file name.
			//Output: a tra_list_t struct.

			
			/*t*/
			printf( "Handling the %i^th trajectory ...\n", i+1);
			printf( "Reading the raw trajectory ...\n");
			/*t*/
			
			raw_data = read_trajectory_v2( i_fp);

			/*
			//Print the tra_list_t struct.
			fprintf( o_fp1, "%i  %i ", tra_cnt, raw_data->pos_n - 1);
			print_tra_list( raw_data, o_fp1);
			fprintf( o_fp1, "\n");
			*/			
		
			//Perform the direction-preserving simplification.
			//Input: raw data, eps.
			//Ouput: a tra_list_t struct.
			/*t*/
			printf( "DPTS ...\n");
			/*t*/
			
			dir_simp_data = dir_simp_trajectory( raw_data, eps);
			
			/*t/
			printf( "Finished!\n");
			/*t*/
			
			//Print the tra_list_t struct.
			fprintf( o_fp1, "%i  %i ", i, dir_simp_data->pos_n);
			print_tra_list( dir_simp_data, o_fp1);
			fprintf( o_fp1, "\n");
		
			//Perfrom the position-preserving simplification.
			//Input: raw data, the quota.
			//Output: a tra_list_ struct.
			//Algorithm Dougolas-Peucker.
			

			//Case 1: Perpendicular distance.
			/*t*/
			printf( "Position-preserving TS (perpendicular distance) ...\n");
			/*t*/
			
			quota = dir_simp_data->pos_n - 1;
			dist_simp_data_per = dist_simp_trajectory( raw_data, quota, 1);
			
			/*t/
			printf( "Finished!\n");
			/*t*/
			
			//Print the tra_list_t struct
			fprintf( o_fp2, "%i  %i ", i, dist_simp_data_per->pos_n);
			print_tra_list( dist_simp_data_per, o_fp2);
			fprintf( o_fp2, "\n");

			//Case 2: Synchronous Euclidean distance.
			/*t*/
			printf( "Position-preserving TS (synchronous Euclidean distance) ...\n");
			/*t*/
			
			quota = dir_simp_data->pos_n - 1;
			dist_simp_data_syn = dist_simp_trajectory( raw_data, quota, 2);
			
			/*t/
			printf( "Finished!\n");
			/*t*/
			
			//Print the tra_list_t struct
			fprintf( o_fp3, "%i  %i ", i, dist_simp_data_syn->pos_n);
			print_tra_list( dist_simp_data_syn, o_fp3);
			fprintf( o_fp3, "\n");
	

			//Release the resources.
			tra_list_release( raw_data);
			tra_list_release( dir_simp_data);
			tra_list_release( dist_simp_data_per);
			tra_list_release( dist_simp_data_syn);

		}//for(i)

		fclose( i_fp);
		fclose( o_fp1);
		fclose( o_fp2);
		fclose( o_fp3);

	}//while( fscanf).

	//
	fclose( c_fp);
}

/*
 *	Calculate the dot-product between two vectors.
 */
double calc_dot_product( loc_t* loc_v11,loc_t* loc_v12, loc_t* loc_v21, loc_t* loc_v22)
{
	return ( loc_v12->x - loc_v11->x) * ( loc_v22->x - loc_v21->x) +
			( loc_v12->y - loc_v11->y) * ( loc_v22->y - loc_v21->y);
}


/*
 *	Calculate the distance between two line segments (loc_v11, loc_v12) and (loc_v21, loc_v22).
 */
double calc_seg_dist( loc_t* loc_v11,loc_t* loc_v12, loc_t* loc_v21, loc_t* loc_v22)
{
	float u_1, u_2, angle_v;
	double leng_1, leng_2, l_1, l_2, l_3, l_4, d_per, d_para, d_the;

	loc_t* p_tmp;
	loc_t p_s, p_e;


	//Decide which segment is longer.
	//Make it be the case that the second segment is longer.
	leng_1 = calc_Euclidean_dist( loc_v11, loc_v12);
	leng_2 = calc_Euclidean_dist( loc_v21, loc_v22);

	if( leng_1 > leng_2)
	{
		//Switch the two segments.
		p_tmp = loc_v11;
		loc_v11 = loc_v21;
		loc_v21 = p_tmp;

		p_tmp = loc_v12;
		loc_v12 = loc_v22;
		loc_v22 = p_tmp;
	}


	//Compute p_s and p_e
	u_1 = calc_dot_product( loc_v21, loc_v11, loc_v21, loc_v22) / pow( leng_2, 2);
	u_2 = calc_dot_product( loc_v21, loc_v12, loc_v21, loc_v22) / pow( leng_2, 2);

	p_s.x = loc_v21->x + u_1 * ( loc_v22->x - loc_v21->x);
	p_s.y = loc_v21->y + u_1 * ( loc_v22->y - loc_v21->y);

	p_e.x = loc_v21->x + u_2 * ( loc_v22->x - loc_v21->x);
	p_e.y = loc_v21->y + u_2 * ( loc_v22->y - loc_v21->y);


	//Compute d_per.
	l_1 = calc_Euclidean_dist( loc_v11, &p_s);
	l_2 = calc_Euclidean_dist( loc_v12, &p_e);

	d_per = ( pow( l_1, 2) + pow( l_2, 2)) / ( l_1 + l_2);

	//Compute d_para.
	l_3 = calc_Euclidean_dist( loc_v21, &p_s);
	l_4 = calc_Euclidean_dist( loc_v22, &p_e);
	d_para = l_3 <= l_4 ? l_3 : l_4;


	//Compute d_the.
	angle_v = calc_angle( loc_v11, loc_v12, loc_v21, loc_v22);
	d_the = sin( angle_v) * leng_2;

	return d_per + d_para + d_the;
}


/*	
 *	Calculate the within-cluster distance.
 *	The cluster is a set of segments.
 */
double calc_within_cluster_dist( seg_t* seg_arr_v, int seg_n)
{
	int i, j;
	double dist;


	dist = 0;
	for( i=0; i<seg_n; i++)
	{
		for( j=0; j<seg_n; j++)
		{
			if( i != j)
			{
				dist += calc_seg_dist( &seg_arr_v[ i].loc_v1, &seg_arr_v[ i].loc_v2, 
										&seg_arr_v[ j].loc_v1, &seg_arr_v[ j].loc_v2);
			}
		}
	}

	return dist;	
}

/*
 *	Calculate the quality measure of a clustering.
 /
double calc_clustering_measure( char* cluster_f_name)
{
	int dim, cluster_n, id, seg_n, i, j;
	double cluster_measure;

	FILE* i_fp;

	seg_t* seg_arr_v;

	if( i_fp = fopen( cluster_f_name)) == NULL)
	{
		fprintf( stderr, "Cannot open %s.\n", cluster_f_name);
		exit( 0);
	}

	fscanf( i_fp, "%i%i", &dim, &cluster_n);

	for( i=0; i<cluster_n; i++)
	{
		fscanf( i_fp, "%i%i", &id, &seg_n);
	
		seg_arr_v = ( seg_t*)malloc( seg_n * sizeof( seg_t));
		memset( seg_arr_v, 0, seg_n * sizeof( seg_t));

		//Read a cluster.
		for( j=0; j<seg_n; j++)
		{
			fscanf( i_fp, "%f%f", seg_arr_v[ j].l
			

}*/

/*
 *
 */
int** collect_similarity_matrix( char* f_name, int &tra_n, double thr, int opt)
{
	int i, j, k, cluster_n, tra_id, cluster_cnt, cluster_id;
	int** cluster_member;
	int** similarity_matrix;

	double** dist, max_dist;

	FILE* i_fp;

	if( ( i_fp = fopen( f_name, "r")) == NULL)
	{
		fprintf( stderr, "Cannot open %s.\n", f_name);
		exit( 0);
	}

	fscanf( i_fp, "%i%i", &tra_n, &cluster_n);

	cluster_member = ( int**)malloc( tra_n * sizeof( int*));
	memset( cluster_member, 0, tra_n * sizeof( int*));

	for( i=0; i<tra_n; i++)
	{
		cluster_member[ i] = ( int*)malloc( cluster_n * sizeof( int));
		memset( cluster_member[ i], 0, cluster_n * sizeof( int));

		//
		fscanf( i_fp, "%i%i", &tra_id, &cluster_cnt);
		for( j=0; j<cluster_cnt; j++)
		{
			fscanf( i_fp, "%i", &cluster_id);

			cluster_member[ i][ cluster_id] ++;
		}
	}


	//Collect the similarity matrix.
	dist = ( double**)malloc( tra_n * sizeof( double*));
	memset( dist, 0, tra_n * sizeof( double*));

	max_dist = 0;
	for( i=0; i<tra_n; i++)
	{
		dist[ i] = ( double*)malloc( tra_n * sizeof( double));
		memset( dist[ i], 0, tra_n * sizeof( double));

		for( j=i+1; j<tra_n; j++)
		{
			//Decide whether trajectory i and j are similar.
			dist[ i][ j] = 0;
			for( k=0; k<cluster_n; k++)
			{
				if( opt == 1)
				{
					//Option 1:
					dist[ i][ j] += pow( cluster_member[ i][ k] - cluster_member[ j][ k], 2);
				}
				else
				{
					//Option 2:
					if( cluster_member[ i][ k] == 0 && cluster_member[ j][ k] != 0 ||
						cluster_member[ i][ k] != 0 && cluster_member[ j][ k] == 0 )
						dist[ i][ j] += 1;
				}
			}

			dist[ i][ j] = sqrt( dist[ i][ j]);

			if( dist[ i][ j] > max_dist)
				max_dist = dist[ i][ j];
		}
	}


	similarity_matrix = ( int**)malloc( tra_n * sizeof( int*));
	memset( similarity_matrix, 0, tra_n * sizeof( int*));
	for( i=0; i<tra_n; i++)
	{		
		similarity_matrix[ i] = ( int*)malloc( tra_n * sizeof( int));
		memset( similarity_matrix[ i], 0, tra_n * sizeof( int));

		for( j=i+1; j<tra_n; j++)
		{
			if( dist[ i][ j] / max_dist <= thr)
				similarity_matrix[ i][ j] = 1;
		}
	}


	//Release the resources.
	for( i=0; i<tra_n; i++)
	{
		free( cluster_member[ i]);
		free( dist[ i]);
	}

	free( cluster_member);
	free( dist);
	fclose( i_fp);

	return similarity_matrix;
}

/*
 *
 */
cluster_measure_t* calc_cluster_measure( int** s_matrix_1, int** s_matrix_2, int size)
{
	int i, j;
	cluster_measure_t* cluster_measure_v;

	cluster_measure_v = ( cluster_measure_t*)malloc( sizeof( cluster_measure_t));
	memset( cluster_measure_v, 0, sizeof( cluster_measure_t));

	for( i=0; i<size; i++)
	{
		for( j=i+1; j<size; j++)
		{
			if( s_matrix_1[ i][ j] != 0 && s_matrix_2[ i][ j] != 0)
				cluster_measure_v->TP ++;
			if( s_matrix_1[ i][ j] == 0 && s_matrix_2[ i][ j] != 0)
				cluster_measure_v->FP ++;
			if( s_matrix_1[ i][ j] != 0 && s_matrix_2[ i][ j] == 0)
				cluster_measure_v->FN ++;
			if( s_matrix_1[ i][ j] == 0 && s_matrix_2[ i][ j] == 0)
				cluster_measure_v->TN ++;
		}
	}

	return cluster_measure_v;
}

/*
 *
 */
void collect_clustering_measures( )
{
	int i, ins_n, tra_n, opt;
	int** s_matrix_raw, **s_matrix_dir, **s_matrix_dist;

	double thr;

	char raw_cluster_f[ MAX_FILENAME_LENG];
	char dir_cluster_f[ MAX_FILENAME_LENG];
	char dist_cluster_f[ MAX_FILENAME_LENG];

	cluster_measure_t* cluster_measure_v_dir, *cluster_measure_v_dist;
	
	FILE* c_fp, *s_fp;

	printf( "Please choose the option:\n\t=1:weighted\n\t=2:un-weighted\n");
	scanf( "%i", &opt);
	
	if( ( c_fp = fopen( CLUSTER_MEASURE_CONFIG, "r")) == NULL ||
		( s_fp = fopen( CLUSTER_MEASURE_STAT, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open %s or %s\n", CLUSTER_MEASURE_CONFIG, CLUSTER_MEASURE_STAT);
		exit( 0);
	}

	//Handle the instances.
	ins_n = 1;
	while( fscanf( c_fp, "%s%s%s%lf", raw_cluster_f, dir_cluster_f, dist_cluster_f, &thr) != EOF)
	{
		printf( "Handling #Case %i ...\n", ins_n ++);

		//Collect the measures.
		s_matrix_raw = collect_similarity_matrix( raw_cluster_f, tra_n, thr, opt);
		s_matrix_dir = collect_similarity_matrix( dir_cluster_f, tra_n, thr, opt);
		s_matrix_dist = collect_similarity_matrix( dist_cluster_f, tra_n, thr, opt);

		cluster_measure_v_dir = calc_cluster_measure( s_matrix_raw, s_matrix_dir, tra_n);
		cluster_measure_v_dist = calc_cluster_measure( s_matrix_raw, s_matrix_dist, tra_n);
		
		//Print the statistics.
		fprintf( s_fp, "%s\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n",
			dir_cluster_f, cluster_measure_v_dir->TP, cluster_measure_v_dir->TN, 
							cluster_measure_v_dir->FP, cluster_measure_v_dir->FN,
							cluster_measure_v_dist->TP, cluster_measure_v_dist->TN, 
							cluster_measure_v_dist->FP, cluster_measure_v_dist->FN);

		//Release the resources.
		for( i=0; i<tra_n; i++)
		{
			free( s_matrix_raw[ i]);
			free( s_matrix_dir[ i]);
			free( s_matrix_dist[ i]);
		}
		
		free( s_matrix_raw);
		free( s_matrix_dir);
		free( s_matrix_dist);

		free( cluster_measure_v_dir);
		free( cluster_measure_v_dist);
	}

	fclose( c_fp);
	fclose( s_fp);
}


#ifndef WIN32
/*
 *	Compress each trajectory in the dataset.
 */
void compress_data_set( )
{
	int d_1, d_2, d_3, i, folder_num, folder_cnt, file_cnt, file_n_thr;
	float ori_pos_cnt, simp_pos_cnt;
	float eps[ 19] = {0, 0.001, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3};

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
	
	for( i = 0; i < 19; i++)
	{
		//Traverse the eps setting.
		ori_pos_cnt = 0;
		simp_pos_cnt = 0;
		
		folder_cnt = 0;

		//
		//for( d_1 = 0; d_1 <= 9; d_1 ++)
		//{
			for( d_2 = 0; d_2 <= 9; d_2 ++)
			{
				for( d_3 = 3; d_3 <= 14; d_3 ++)
				{
					folder_cnt ++;
					if( folder_cnt > folder_num)
						goto E;

					//sprintf( folder_name, "%i%i%i", d_1, d_2, d_3);
					//sprintf( dir_path, "%s%s%s", "./data/Geolife/Data/", folder_name, "/Trajectory/");

					sprintf( folder_name, "%i%i", d_2, d_3);
					sprintf( dir_path, "%s%s%s", "./data/T-Drive/", folder_name, "/");
					
					/*t*/
					printf( "%s\t%f\n", dir_path, eps[ i]);
					/*t*/
					
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

							ori_pos_cnt += tra_list_v->pos_n;

							//Perform the simplification.
							simp_pos_cnt += DPTS( tra_list_v, eps[ i], 5);

							//Release the resources.
							tra_list_release( tra_list_v);																			
						}
						
					}//while( dir_r)		
					
					closedir( dir_p);
				}//for(d_3)
			}//for( d_2)
		//}//for( d_1)
E:
		fprintf( s_fp, "%f\t%f\t%f\t%f\n", eps[ i], ori_pos_cnt, simp_pos_cnt, simp_pos_cnt / ori_pos_cnt);
	}//for( i)

	fclose( s_fp);		
}

/*
 *	Collect the statistics of the datasets.
 */
void collect_statistics( )
{
	int d_1, d_2, d_3, i, folder_num, folder_cnt, file_cnt, file_n_thr;
//	float ori_pos_cnt, simp_pos_cnt;
//	float eps[ 19] = {0, 0.001, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3};

	char folder_name[ MAX_FILENAME_LENG];
	char dir_path[ MAX_PATHNAME_LENG];
	char f_name[ MAX_FILENAME_LENG];

	DIR* dir_p;
	struct dirent* dir_r;
	unsigned char is_file =0x8;

	int tra_n_cnt, pos_n_cnt, dir_n_cnt, tra_n_estimate, pos_n_estimate;
	int* pos_n_array;
	float* dir_array;
	float pos_n_mean, pos_n_div, dir_mean, dir_div;
	double dir_sum;
	float angle_v;
	

	tra_list_t* tra_list_v;
	//tra_node_t* tra_node_v1, *tra_node_v2, *tra_node_v3;
	R_list_t* R_list_v;
	R_node_t* R_node_v1, *R_node_v2;

	FILE *c_fp, *s_fp;

	if( ( c_fp = fopen( DATA_CONFIG_FILE, "r")) == NULL ||
		( s_fp = fopen( DATA_STATISTICS_FILE, "w")) == NULL)
	{
		fprintf( stderr, "Cannot open the config/stat files [collect_statistics].\n");
		exit( 0);
	}

	/*t*/
	//printf( "# of folders to simplify:\n");
	fscanf( c_fp, "%i", &folder_num);

	//
	fscanf( c_fp, "%i", &file_n_thr);

	//printf( "# of trajectories/files:\n");
	fscanf( c_fp, "%i", &tra_n_estimate);

	fscanf( c_fp, "%i", &pos_n_estimate);
//	printf( "the threshold of the # of files:\n");
//	scanf( "%i", &file_n_thr);
	/*t*/

	pos_n_array = ( int*)malloc( sizeof( int) * tra_n_estimate);
	memset( pos_n_array, 0, sizeof( int) * tra_n_estimate);

	dir_array = ( float*)malloc( sizeof( float) * pos_n_estimate);
	memset( dir_array, 0, sizeof( float) * pos_n_estimate);
	
	//for( i = 0; i < 19; i++)
	//{
		//Traverse the eps setting.
	//	ori_pos_cnt = 0;
	//	simp_pos_cnt = 0;
		
		folder_cnt = 0;

		tra_n_cnt = 0;
		pos_n_cnt = 0;
		
		dir_n_cnt = 0;
		dir_sum = 0;


		//
//		for( d_1 = 0; d_1 <= 9; d_1 ++)
//		{
			for( d_2 = 0; d_2 <= 9; d_2 ++)
			{
				for( d_3 = 1; d_3 <= 14; d_3 ++)
				{
					folder_cnt ++;
					if( folder_cnt > folder_num)
						goto E;

//					sprintf( folder_name, "%i%i%i", d_1, d_2, d_3);
//					sprintf( dir_path, "%s%s%s", "./data/Geolife/Data/", folder_name, "/Trajectory/");

					sprintf( folder_name, "%i%i", d_2, d_3);
					sprintf( dir_path, "%s%s%s", "./data/T-Drive/", folder_name, "/");
					
					/*t*/
					printf( "%s\n", dir_path);
					/*t*/
					
					//Handle the current folder.
					if( ( dir_p = opendir( dir_path)) == NULL)
					{
						fprintf( stderr, "Cannot open %s.\n", dir_path);
						exit( 0);
					}
					
					while( ( dir_r = readdir( dir_p)) != NULL)
					{
						//
						if( dir_r->d_type == is_file)
						{
		//					file_cnt ++;
					//		if( file_cnt > file_n_thr)
					//			break;

							//tra_cnt ++;

							//
						//	if( tra_n_cnt > file_n_thr)
						//		goto E;

							sprintf( f_name, "%s%s", dir_path, dir_r->d_name);

							/*t*/
							printf( "%s\n", f_name);
							/*t*/
						
							//Read the trajectory.
							tra_list_v = read_trajectory_v3( f_name);

							//pos_n.
							pos_n_array[ tra_n_cnt++] = tra_list_v->pos_n;
							pos_n_cnt += tra_list_v->pos_n;
							
							//dir.
							/*
							if( tra_list_v->pos_n > 2)
							{
								tra_node_v1 = tra_list_v->head->next;
								tra_node_v2 = tra_node_v1->next;
								tra_node_v3 = tra_node_v2->next;

								while( tra_node_v3 != NULL)
								{
									angle_v = calc_angle( &tra_node_v1->tri_v.loc_v, &tra_node_v2->tri_v.loc_v,
															&tra_node_v2->tri_v.loc_v, &tra_node_v3->tri_v.loc_v);

									//printf( "%f\n", angle_v);
									dir_array[ dir_n_cnt++] = angle_v;

									dir_sum += angle_v;

									tra_node_v1 = tra_node_v2;
									tra_node_v2 = tra_node_v3;
									tra_node_v3 = tra_node_v3->next;
								}
							}
							*/
							//
							if( tra_list_v->pos_n > 2)
							{
								R_list_v = R_list_transform( tra_list_v);
							
								R_node_v1 = R_list_v->head->next;
								R_node_v2 = R_node_v1->next;
								while( R_node_v2 != NULL)
								{
									angle_v = R_node_v1->slope - R_node_v2->slope;

									//if( errno)
										//printf( "errno = %i\n", errno);

									//angle_v = angle_v >= 0 ? angle_v : -angle_v;
									if( angle_v < 0)
										angle_v = - angle_v;

									if( angle_v > 2 * PI - angle_v)
										angle_v = 2 * PI - angle_v;

									//angle_v = angle_v <= 2 * PI - angle_v ? angle_v : 2 * PI - angle_v;

									//printf( "%f\n", angle_v);

									if( angle_v >= 0 && angle_v < PI)
									{
										dir_sum += angle_v;
										dir_array[ dir_n_cnt++] = angle_v;
									}

									//
									//printf( "dir_sum: %lf\n", dir_sum);

									R_node_v1 = R_node_v2;
									R_node_v2 = R_node_v2->next;
								}
							}
							//
							//printf( "dir_sum: %lf\n", dir_sum);

						
							//ori_pos_cnt += tra_list_v->pos_n;

							//Perform the simplification.
							//simp_pos_cnt += DPTS( tra_list_v, eps[ i], 5);

							//Release the resources.
							if( tra_list_v->pos_n > 2)
								R_list_release( R_list_v);
							tra_list_release( tra_list_v);																			
						}
						
					}//while( dir_r)		
					
					closedir( dir_p);
				}//for(d_3)
			}//for( d_2)
		//}//for( d_1)
E:
		//
		pos_n_mean = ( float)pos_n_cnt / tra_n_cnt;

		pos_n_div = 0;
		for( i=0; i<tra_n_cnt; i++)
		{
			pos_n_div += pow( pos_n_array[ i] - pos_n_mean, 2);
		}

		pos_n_div /= tra_n_cnt;

		pos_n_div = sqrt( pos_n_div);
			
		
		//
		printf( "dir_sum: %f\n", dir_sum);
		if( dir_n_cnt != 0)
			dir_mean = dir_sum / dir_n_cnt;
		else
			dir_mean = 0;

		dir_div = 0;
		for( i=0; i<dir_n_cnt; i++)
		{
			dir_div += pow( dir_array[ i] - dir_mean, 2);
		}

		if( dir_n_cnt != 0)
		{
			dir_div /= dir_n_cnt;
			dir_div = sqrt( dir_div);
		}
		else
			dir_div = 0;

		//
		fprintf( s_fp, "%i\n%i\n%f\n%f\n\n", pos_n_cnt, tra_n_cnt, pos_n_mean, pos_n_div);
		fprintf( s_fp, "%f\n%i\n%f\n%f\n\n", dir_sum, dir_n_cnt, dir_mean, dir_div);

		fprintf( stdout, "%i\n%i\n%f\n%f\n\n", pos_n_cnt, tra_n_cnt, pos_n_mean, pos_n_div);
		fprintf( stdout, "%lf\n%i\n%f\n%f\n\n", dir_sum, dir_n_cnt, dir_mean, dir_div);


//	}//for( i)
	
	free( pos_n_array);
	free( dir_array);
	fclose( c_fp);
	fclose( s_fp);		
}

#endif

