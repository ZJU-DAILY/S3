/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



#ifndef MIN_ERROR_H
#define MIN_ERROR_H


#include "RN_appro.h"
#include "wavelet.h"

#define TRA_SIZE_TRUNCATION_OPT	1

extern emp_stat_t emp_stat_v;
extern LOC_TYPE precision_thr_v2;

//Starting from 17 July, 2013.
//New APIs for the Min-Error problem.

error_space_t* error_space_alloc( int pos_n);

void error_space_release( error_space_t* error_space_v);

//The DP algorithm.
error_space_t* const_error_space_v1( R_list_t* R_list_v);

float find_closest_slope( bst_t* bst_v, float slope_v);

float calc_max_angular_diff( bst_t* bst_v, float slope_v);

error_space_t* const_error_space_v2( R_list_t* R_list_v);

float DP_Error( R_list_t* R_list_v, int W, int ver_opt);

//The Error-Search algorithm.
//int error_affordability_check( R_list_t* R_list_v, float eps);

float* reform_error_space( error_space_t* error_space_v);

id_list_t* Error_Search( R_list_t* R_list_v, int W);

//The Span-Search algorithm.

//span space maintenance.
span_space_t* span_space_alloc( int pos_n);

void span_space_release( span_space_t* span_space_v);

void add_span_space_entry( span_space_t* span_space_v, int s, int e, int j);

span_space_t* const_span_space( R_list_t* R_list_v);

//pivot finding.
float get_median( span_space_t* span_space_v, array_node_t* array_node_v);

int	is_a_pivot( span_space_t* span_space_v, float span_v);

float find_pivot( span_space_t* span_space_v);

//span affordability check.
//angular_range_t* calc_mcar( bst_t* bst_v);

//float calc_span_of_mcar( bst_t* bst_v);

id_list_t* span_affordability_check( R_list_t* R_list_v, float span_v);

float get_matrix_entry( span_space_t* span_space_v, int i, int j);

void prune_span_space( span_space_t* span_space_v, float pivot, int tag);

float comp_simp_error( R_list_t* R_list_v, id_list_t* id_list_v);

id_list_t* Span_Search( R_list_t* R_list_v, int W);

//Baselines of approximate algorithm.
//Adapted from those baselines for the Min-Size problem.
//int Split_sub( R_array_t* R_array_v, int seg_sta, int seg_end, int W);

//int Split( R_list_t* R_list_v, int W);

//id_list_t* Merge_sub( R_array_t* R_array_v, int W);

id_list_t* Merge_sub2( R_array_t* R_array_v, int W);

id_list_t* Merge( R_list_t* R_list_v, int W);

//id_list_t* Greedy_v1( R_list_t* R_list_v, int W);

//id_list_t* Greedy_v2( R_list_t* R_list_v, int W);

void ini_part_node_dist( part_node_t* part_node_v);

id_list_t* Dougolas_Peucker_adapt( R_list_t* R_list_v, int W);

float DPTS_v2( tra_list_t* tra_list_v, int W, int alg_opt);

void emp_DPTS_v2( );


//Testing.
void print_error_space( error_space_t* error_space_v);

void print_span_space( span_space_t* span_space_v);

#ifndef WIN32
void compress_data_set_v2( );
#endif


//Added on April 17, 2014.
//Trajectory simplification interface based on HWT.
float emp_TS_HWT( tra_list_t* tra_list_v, int W);

float emp_TS_HWT_v2( tra_list_t* tra_list_v, int W);

wavelet_data_t* TS_HWT( wavelet_data_t* wavelet_data_v, int k);

float calc_error_tra( tra_list_t* tra_list_v1, tra_list_t* tra_list_v2);

float calc_error_R( R_list_t* R_list_v1, R_list_t* R_list_v2);

//Trajectory simplification interface with the feasibility check component done by the Douglas-Peucker algorithm.
float Error_Search_DouglasPeucker( R_list_t* R_list_v, int W);

#endif