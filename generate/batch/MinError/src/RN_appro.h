/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */


/*
 *	1. read_data requires the graph to be un-directed graph and
 *	each edge is represented by only one pair.
 */

#ifndef RN_APPRO_H
#define	RN_APPRO_H

#include "data_struct.h"
#include "data_utility.h"
#include "bst.h"
#include "b_heap.h"

#include "wavelet.h"

#define PERPENDICULAR_OPTION	2

extern int	MIN_HEAP_OPT;


#define min( a, b)	( ( a) < ( b) ? ( a) : ( b))
#define max( a, b)	( ( a) > ( b) ? ( a) : ( b))
#define abs_f( a)	( ( a) >= 0 ? ( a) : ( -( a)))

//Supporting APIs.

R_set_t* R_set_alloc( );

void R_set_release( R_set_t* R_set_v);

R_list_t* R_list_const( RN_graph_t* RN_graph_v, int sta_id, RN_graph_list_t* g_list_v);

R_set_t* RN_decompose( RN_graph_t* RN_graph_v);

void range_decompose( slope_r_t* slope_r_v, slope_r_t* & slope_r_v1, slope_r_t* & slope_r_v2);

//void range_intersect( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2, slope_r_t* & slope_r_v);

//void append_common_range( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2, slope_r_list_t* slope_r_list_v);

//slope_r_list_t* calc_common_range( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2);

float calc_angle( loc_t* loc_v11, loc_t* loc_v12, loc_t* loc_v21, loc_t* loc_v22);

float calc_angular_diff( float slope_v1, float slope_v2);

float calc_error( R_node_t* R_node_v1, R_node_t* R_node_v2);

int check_error( R_node_t* R_node_v1, R_node_t* R_node_v2, float eps);

float calc_leng( loc_t* loc_v1, loc_t* loc_v2);


//Min-Number.

int** calc_appr_table1( R_list_t* R_list_v, float eps);

int DP_Number( R_list_t* R_list_v, float eps, int ver_opt);

eps_g_t* eps_g_alloc( int vertex_n);

void eps_g_release( eps_g_t* eps_g_v);

eps_g_t* eps_g_const( R_list_t* R_list_v, float eps, int opt);

eps_g_t* eps_g_const1( R_list_t* R_list_v, float eps);

eps_g_t* eps_g_const2( R_list_t* R_list_v, float eps);

int calc_SP1( eps_g_t* eps_g_v, int s_id, int d_id);

int SP_Method_sub( R_list_t* R_list_v, float eps, int ver_opt);

int Min_Number_sub( R_set_t* R_set_v, float eps, int opt);

int Min_Number( RN_graph_t* RN_graph_v, float eps, int opt);


//Min-Error.

int compare (const void * a, const void * b);

float* search_space_R( R_list_t* R_list_v);

float Search_Error_R( R_list_t* R_list_v, int L);		//

float DP_Error_R_sub( R_node_t* R_node_v1, R_node_t* R_node_v2, int L);	//Recursive implementation.

float DP_Error_R( R_list_t* R_list_v, int L);

search_space_t* search_space_N( RN_graph_t* RN_graph_v);

float Search_Error_N( RN_graph_t* RN_graph_v, int L, int check_opt);

float DP_Error_N_sub( R_list_t* R_list_v1, R_list_t* R_list_v2, int num, int L, int opt);

float DP_Error_N( RN_graph_t* RN_graph_v, int L, int check_opt);

float error_retrieve( b_heap_t* b_h, int* tag);

float Min_Error( RN_graph_t* RN_graph_v, int L, int alg_opt);


//Testing functions.

void print_R_set( R_set_t* R_set_v);

void print_eps_g( eps_g_t* eps_g_v);

void print_RN_graph( RN_graph_t* RN_graph_v);

void print_appr_table( int** appr_table, int row_n, int col_n);

void print_search_space( search_space_t* search_space_v);

void test( );


//----------------------Modified on 26, Feb 2012-----------------------

/*
 *	Empirical Study Part 1 [Verification of the algorithms in the paper].
 *		a. Min-Number
 *		b. Min-Error
 */

//Min-Number.

//slope_r_t* anlge_range_decompose( slope_r_t* slope_r_v);

float modulo_op( float t, float modulo); 

int	is_in_slope_r( float t, float a, float b);

void slope_range_intersect( slope_r_t* slope_r_v1, slope_r_t* slope_r_v2, slope_r_t* &slope_r_v3, slope_r_t* &slope_r_v4);

slope_r_list* slope_r_list_alloc( );

void append_slope_r_list_entry( slope_r_list_t* slope_r_list_v, slope_r_t slope_r_v);

void slope_r_list_release( slope_r_list_t* slope_r_list_v);

slope_r_list_t* range_intersect( slope_r_list_t* slope_r_list_v, slope_r_t* slope_r_v);

int is_in_slope_r_list( slope_r_list_t* slope_r_list_v, float slope);

int** calc_appr_table2( R_list_t* R_list_v, float eps);


//Min-Error (Construct the search space).

KEY_TYPE closest_gap( bst_t* T, KEY_TYPE key);

float calc_error_boundary( float b_1, float b_2, float slope);

//span_m_t* span_m_aloc( );

void span_m_ini( span_m_t* span_m_v);

void span_m_update( span_m_t* span_m_v, float slope);

slope_r_t* calc_span( span_m_t* span_m_v);

float calc_error_span( bst_t* T, span_m_t* span_m_v, float slope);

//float* search_space_R2( R_list_t* R_list_v);

//search_space_t* search_space_N2( RN_graph_t* RN_graph_v);

float* search_space_R3( R_list_t* R_list_v);

search_space_t* search_space_N3( R_set_t* R_set_v);


//Approximation methods for Min-Error: Merge_Error and Merge_Span.

float Merge_Error( RN_graph_t* RN_graph_v, int L);

span_m_t* span_combine( span_m_t* span_m_v1, span_m_t* span_m_v2);

float Merge_Span( RN_graph_t* RN_graph_v, int L);

b_heap_t* Merge_Span2( RN_graph_t* RN_graph_v, int L, int* &tag);

void test2( );

/*
 *	Empirical Study Part 2: [Verification the following two usages of road network approximation].
 *		a. Efficiency improvement.
 *		b. Protection of the shortest path distance.
 */

int calc_SP_RN( RN_graph_t* RN_graph_v_sim, RN_graph_t* RN_graph_v_ori, eps_g_t* eps_g_v, int s_id, int d_id);

RN_graph_t* RN_simplify( RN_graph_t* RN_graph_v, float eps, int opt);

float dijkstra( RN_graph_t* RN_graph_v, int s_id, int d_id);

void RN_simplify_verify( RN_graph_t* RN_graph_v_ori, RN_graph_t* RN_graph_v_sim, int q_times);

void test_RN_simplify( );

/*
 *	Empirical Study Part 3: [Verification that other error mechanisms cannot protect the direction information].
 *		a. distance-oriented.
 *		b. length-oriented.
 */

/*
float DP_Error_Distance_R( RN_graph_t* RN_graph_v, int L);

float DP_Error_Distance_N( RN_graph_t* RN_graph_v, int L);

float Min_Error_Distance( RN_graph_t* RN_graph_v, int L, int alg_opt);

float DP_Error_Length_R( RN_graph_t* RN_graph_v, int L);

float DP_Error_Length_N( RN_graph_t* RN_graph_v, int L);

float Min_Error_Length( RN_graph_t* RN_graph_v, int L, int alg_opt);
*/


//Updated on 21 January, 2013 for Min-Number.
//
void calc_fdr_segment( slope_r_t* slope_r_v, R_node_t* R_node_v, float eps);

id_list_t* id_list_alloc( );

void append_id_list_entry( id_list_t* id_list_v, int pos_id);

void ahead_id_list_entry( id_list_t* id_list_v, int pos_id);

void id_list_print( id_list_t* id_list_v);

void id_list_release( id_list_t* id_list_v);

id_list_t* calc_SP2( eps_g_t* eps_g_v, int s_id, int d_id);

tra_list_t* tra_const_from_id_list( tra_list_t* ori_tra_list_v, id_list_t* id_list_v);

fdr_array_t* fdr_array_alloc( int seg_n);

void fdr_array_release( fdr_array_t* fdr_array_v);

fdr_array_t* fdr_maintenance( R_list_t* R_list_v, float eps);

R_array_t* R_array_alloc( int seg_n);

void R_array_release( R_array_t* R_array_v);

R_array_t* R_array_transform( R_list_t* R_list_v);

void fdr_maintenance_on_demand( fdr_array_t* fdr_array_v, R_array_t* R_array_v, float eps, int seg_sta, int seg_end);

int check_error_fdr( fdr_array_t* fdr_array_v, R_array_t* R_array_v, float eps, int pos_sta, int pos_end);

hop_list_t* hop_list_alloc( );

void ahead_hop_list_entry( hop_list_t* hop_list_v, int pos_id, R_node_t* seg_before, R_node_t* seg_after);

void append_hop_list_entry( hop_list_t* hop_list_v, int pos_id, R_node_t* seg_before, R_node_t* seg_after);

void hop_list_release( hop_list_t* hop_list_v);

//int SP_Method_prac_num( R_list_t* R_list_v, float eps, int fdr_opt);

//id_list_t* SP_Method_sub2( R_list_t* R_list_v, float eps, int fdr_opt);

int SP_Method_B( R_list_t* R_list_v, float eps);

int SP_Method_theo( R_list_t* R_list_v, float eps);

id_list_t* SP_Method_prac( R_list_t* R_list_v, float eps);

id_list_t* SP_Method_prac_and_theo( R_list_t* R_list_v, float eps);

//Approximate algorithms.

int Split_sub( R_array_t* R_array_v, int seg_sta, int seg_end, float eps);

int Split( R_list_t* R_list_v, float eps);

group_list_t* group_list_alloc( );

void append_group_list_entry( group_list_t* group_list_v, int sta_pos_id, int end_pos_id);

//void group_list_merge( group_node_t* group_node_v);

void group_list_release( group_list_t* group_list_v);

id_list_t* Merge_sub( R_array_t* R_array_v, float eps);

id_list_t* Merge_sub2( R_array_t* R_array_v, float eps);

id_list_t* Merge( R_list_t* R_list_v, float eps);

id_list_t* Greedy_v1( R_list_t* R_list_v, float eps);

id_list_t* Greedy_v2( R_list_t* R_list_v, float eps);

id_list_t* Intersect( R_list_t* R_list_v, float eps);


//For collecting error statistics.
//
void collect_length_and_speed_error( R_list_t* R_list_v_ori, R_list_t* R_list_v_appr, float* length_error, float* speed_error);

float collect_position_error_bound( R_list_t* R_list_v_appr, float eps, int dist_opt);

void collect_position_error( R_list_t* R_list_v_ori, R_list_t* R_list_v_appr, float* position_error, int dist_opt);

void collect_direction_error( R_list_t* R_list_v_ori, R_list_t* R_list_v_appr, float* direction_error);


//For testing only.
//
void print_error_matrix( R_list_t* R_list_v, int p_opt);

void print_hop_list( hop_list_t* hop_list_v);

//void print_slope_r( slope_r_t* slope_r_v);

void print_slope_r_list( slope_r_list_t* slope_r_list_v);

//Empirical study.
int DPTS( tra_list_t* tra_list_v, float eps, int alg_opt);

void emp_DPTS( );


//For the clustering application.
//Prepare the clustering datasets.
tra_list_t* read_trajectory_v1( char* f_name);

tra_list_t* read_trajectory_v2( FILE* i_fp);

tra_list_t* read_trajectory_v3( char* f_name);

tra_list_t* retrieve_trajectory( tra_list_t* raw_data, id_list_t* id_list_v);

tra_list_t* dir_simp_trajectory( tra_list_t* raw_data, float eps);

part_list_t* part_list_alloc( );

void part_list_release( part_list_t* part_list_v);

double calc_per_dist( loc_t* loc_v1, loc_t* seg_end1, loc_t* seg_end2, int opt);

double calc_syn_Euclidean_dist( loc_t* loc_v1, loc_t* seg_end1, loc_t* seg_end2, float time_portion);

void ini_part_node_dist( part_node_t* part_node_v, int dist_opt);

id_list_t* position_TS( R_list_t* R_list_v, int quota, int dist_opt);

DIST_TYPE calc_error_dist( R_node_t* R_node_v1, R_node_t* R_node_v2, int dist_opt);

DIST_TYPE position_TS_DP( R_list_t* R_list_v, int quota, int dist_opt);

id_list_t* SP_Method_prac_dist( R_list_t* R_list_v, float eps, int dist_opt);

tra_list_t* dist_simp_trajectory( tra_list_t* raw_data, int quota, int dist_opt);

void print_tra_list( tra_list_t* tra_list_v, FILE* o_fp);

#ifndef WIN32
void prepare_cluster_data_v1( );
#endif

void prepare_cluster_data_v2( );

//Performing clustering.
//Use existing clustering algorithms.
//Write a batch algorithm.
//Based on the configuration file for "Preparing the clustering data".
//Parameter fitting (eps, minPoints)


//Collect the clustering measures.
//Computation function.

double calc_dot_product( loc_t* loc_v11,loc_t* loc_v12, loc_t* loc_v21, loc_t* loc_v22);

double calc_seg_dist( loc_t* loc_v11,loc_t* loc_v12, loc_t* loc_v21, loc_t* loc_v22);

double calc_within_cluster_dist( seg_t* seg_arr_v, int seg_n);

double calc_clustering_measure( char* cluster_f_name);

int** collect_similarity_matrix( char* f_name, int &tra_n, double thr, int opt);

cluster_measure_t* calc_cluster_measure( int** s_matrix_1, int** s_matrix_2, int size);

void collect_clustering_measures( );


#ifndef WIN32
//Collecting compression ratio.
void compress_data_set( );

void collect_statistics( );
#endif

#endif

