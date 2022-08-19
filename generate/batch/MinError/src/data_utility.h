/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



#ifndef	DATA_UTILITY_H
#define DATA_UTILITY_H


#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "data_struct.h"

#ifndef WIN32
#include <sys/types.h>
#include <dirent.h>
#endif

#define	PI	3.14159265

#define		MAX_FILENAME_LENG	256
#define		MAX_LINE_LENG		1024				//Need further confirmation here.
#define		CONFIG_FILE			"config.txt"
#define		STATISTIC_FILE		"stat.txt"
#define		RES_FILE		"res.txt"

#define		REAL_DATA_CONFIG	"real_d_config.txt"

#define		NON_PASSING_CONFIG	"non_passing_config.txt"
#define		NON_PASSING_STAT	"non_passing_stat.txt"


#define		TEMP_FILE			"temp.txt"

#define		COMBINE_FILE_CONFIG	"comb_files_config.txt"
#define		FILE_NAMES_TEMP		"files_temp.txt"
#define		COMBINE_FILE_STAT	"comb_files_stat.txt"

#define		CLUSTER_DATA_CONFIG "cluster_data_config.txt"
#define		CLUSTER_DATA_STAT	"cluster_data_stat.txt"
#define		CLUSTER_DATA_LOC	"./data/cluster_data/"

#define		CLUSTER_DATA_CONFIG_V2	"cluster_data_config_v2.txt"


#define		CLUSTER_MEASURE_CONFIG	"cluster_measure_config.txt"
#define		CLUSTER_MEASURE_STAT	"cluster_measure_stat.txt"


#define		COMPRESSION_STAT		"compression_stat.txt"

#define		DATA_CONFIG_FILE		"data_config.txt"
#define		DATA_STATISTICS_FILE	"data_statistics.txt"

#define		DATA_CONFIG_FILE_V2		"data_config_v2.txt"

#define		WAVELET_CONFIG_FILE		"wavelet_config.txt"

#define		MAX_PATHNAME_LENG	256
#define		MAX_STR_LENG		1000

#define		NUM_LINES_IGNORE	6

#define		PRECISION_THR		0.00001

#define		BATCH_NUMBER		20000000
#define		SIZE_THR			2000000


extern	LOC_TYPE precision_thr_v2;


/*The structure for storing the configuration information.*/
typedef struct config
{
	//Road network.
	char	vertex_f[ MAX_FILENAME_LENG];
	char	edge_f[ MAX_FILENAME_LENG];
	int		vertex_n;
	int		edge_n;
	
	//Parameters.
	float	eps;
	int		L;
	float	L_percent;

//For Empirical Study Part 1.
	int		pro_opt;		//Problem indicator.
	/*
	 *	pro_opt = 1: Min-Number;	
	 *	pro_opt = 2: Min-Error.
	 */

	int		alg_opt;		//Method indicator.
	/*
	 *	Case 1: pro_tag = 1.
	 *		alg_opt = 1:	DP-Number.
	 *		alg_opt = 2:	SP-Method
	 *		alg_opt >= 3:	Approximation methods (if any)
	 *
	 *	Case 2: pro_tag = 2;
	 *		alg_opt = 1:	DP-Error (Search-Error-Road).
	 *		alg_opt = 2:	Search-Error (SP-Method).
	 *		alg_opt = 3:	Merge-Error.
	 *		alg_opt = 4:	Merge-Span.
	 *		alg_opt = 5:	DP-Error (DP-Error-Road).
	 *		alg_opt = 6:	Search-Error (DP-Number).
	 */	

//For Empirical Study Part 2.
	int		q_times;

//For Empirical Study Part 3.
	
}	config_t;


/*The structure for storing the data preprocessing configuration.*/
typedef struct d_config
{
	char input_f[ MAX_FILENAME_LENG];
	char output_f[ MAX_FILENAME_LENG];

	int	 file_opt;

}	d_config_t;


/*The triplet structure.*/
typedef struct quadruple
{
	float	min;
	float	max;

	float	aver;
	float	sum;

}	quadruple_t;

/*The structure for storing the statistics information.*/
typedef struct stat
{
//For Empirical Study Part 1.
	float	r_time;
	float	memory_max;
	float	memory_v;

	int		min_number;
	float	min_error;

	quadruple_t	C;
	int			C_cnt;

	//For Merge_Span2.
	float	error_calc_time;

//For Empirical Study Part 2.
	int		sim_vertex_n;
	int		sim_edge_n;

	float	ori_query_t;
	float	sim_query_t;

	float	ori_memory_max;
	float	ori_memory_v;
	float	sim_memory_max;
	float	sim_memory_v;

	quadruple_t	length_err;

//For Empirical Study Part 3.
	float	min_error_dir;
	float	min_error_dist;
	float	min_error_leng;

}	stat_t;



typedef struct emp_config
{

	//data specific.
	char	f_name[ MAX_FILENAME_LENG];

	int		pos_n;

	int		line_ignore_n;

	float	precision_thr;


	//problem specific.
	float	eps;			//for the Min-Size problem
	int		W;				//for the Min-Error problem
	float	W_percent;	

	//algorithm specific.
	int		alg_opt;

	//appr_bound tag
	int		appr_b_tag;	


	int		dataset_tag;	

}	emp_config_t;


typedef struct emp_stat
{
	float		r_time;
	float		memory_v;
	float		memory_max;

	float		cap_n;			//the number of intersection operations.
	quadruple	C;				//the statistics of C;

	float		cmp_ratio;		//compression ratio;

	float		error;

	//for the practical enhancement;
	float		eps_n;
	float		percent;

	//for approximate algorithms;
	float		appro_f;		

	//for other error measurements.
	float		length_err;
	float		speed_err;
	float		length_err_bound;
	float		speed_err_bound;

	float		pos_err_per;
	float		pos_err_syn;	
	float		pos_err_per_bound;
	float		pos_err_syn_bound;
	
	float		pos_err_per_opti;
	float		pos_err_syn_opti;

	float		pos_err_per_heur;
	float		pos_err_syn_heur;

	float		dir_err_dpts;
	float		dir_err_per_opti;
	float		dir_err_syn_opti;

	float		dir_err_per_heur;
	float		dir_err_syn_heur;

	//float		dir_err_per_bound;
	//float		dir_err_syn_bound;

	//For wavelet transformation;
	float		error_aver;

}	emp_stat_t;


//Added on April 21, 2014;
typedef struct wavelet_config
{
	int		n;
	int		k;
	
	char	f_name_in[ MAX_FILENAME_LENG];
	char	f_name_out[ MAX_FILENAME_LENG];

	char	coeff_f_name_out[ MAX_FILENAME_LENG];

}	wavelet_config_t;

	

extern stat_t stat_v;
extern emp_stat_t emp_stat_v;

#ifndef WIN32
#include <sys/resource.h>

void get_cur_time( rusage* cur_t);

void collect_time( struct rusage* sta_t, struct rusage* end_t, float* user_t);	//, float* system_t);
#endif

double calc_Euclidean_dist( loc_t* loc_v1, loc_t* loc_v2);

void real_data_preprocess( );

config_t* read_config( );

RN_graph_t* RN_graph_alloc( int vertex_n);

void RN_graph_release( RN_graph_t* RN_graph_v);

float calc_abs_slope( loc_t* loc_v1, loc_t* loc_v2);

RN_graph_t* read_data( config_t* config_v);

RN_graph_t* read_data_partial( int vertex_n, char* f_name);

void collect_non_passing_stat( );


//Updated on 27 Jan 2013.
emp_config_t* emp_read_config( );

tra_list_t* tra_list_alloc( );

void append_tra_list_entry( tra_list_t* tra_list_v, triplet_t tri_v);

void tra_list_release( tra_list_t* tra_list_v);

void R_node_ini( R_node_t* R_node_v, tra_node_t* tra_node_v1, tra_node_t* tra_node_v2);

R_list_t* R_list_alloc( );

void append_R_list_entry( R_list_t* R_list_v, R_node_t* R_node_v);

void append_R_list_entry_tra( R_list_t* R_list_v, tra_node_t* tra_node_v1, tra_node_t* tra_node_v2);

R_list_t* R_list_transform( tra_list_t* tra_list_v);

R_list_t* R_list_transform_appr( tra_list_t* tra_list_v, id_list_t* id_list_v);

void R_list_release( R_list_t* R_list_v);

tra_list_t*	emp_read_data_v1( emp_config_t* emp_config_v);

tra_list_t*	emp_read_data_v2( emp_config_t* emp_config_v);

//Real data preprocessing.
int	str_cmp( const void* str_1, const void* str_2);

int combine_files_sub( char* dir_path, char** file_names, int file_n, char* output_f, int size_thr);

int collect_size( char* dir_path, char** file_names, int file_n, char* output_f);

#ifndef WIN32
void combine_files( int func_opt);
#endif

//starting from 17 July, 2013: the Min-Error problem.
//emp_config_t* emp_read_config_v2( );


//Added on April 17, 2014.
//For Haar Wavelet Transformation (HWT)

/*
 *	Implementation of the log function base 2.
 */
int log2( int n);

/*
 *	Check whether a number is a number of power 2.
 */
int	check_power2( int n);

/*
 *	Compute the largest number that is a number of power 2 and bounded by @n.
 */
int comp_bounded_power2( int n);

//tra_list_t* tra_list_const( trace_data_t* trace_data_v);


//trace_data_t* trace_data_alloc( int size);

//trace_data_t* trace_data_const( tra_list_t* tra_list_v);

//trace_data_t* trace_data_const_2( LOC_TYPE* data_v1, LOC_TYPE* data_v2, int size);

//void trace_data_release( trace_data_t* trace_data_v);

void tra_list_to_wavelet_data( tra_list_t* tra_list_v, wavelet_data_t* &wavelet_data_v1, wavelet_data_t* &wavelet_data_v2);

void wavelet_data_to_tra_list( wavelet_data_t* wavelet_data_v1, wavelet_data_t* wavelet_data_v2, tra_list_t* &tra_list_v);

void R_list_to_wavelet_data( R_list_t* R_list_v, wavelet_data_t* &wavelet_data_v1, wavelet_data_t* &wavelet_data_v2);

void wavelet_data_to_R_list( wavelet_data_t* wavelet_data_v1, wavelet_data_t* wavelet_data_v2, R_list_t* &R_list_v);


wavelet_data_t* wavelet_data_alloc( int n);

//wavelet_data_t* wavelet_data_const( LOC_TYPE* data_v, int size);

void wavelet_data_release( wavelet_data_t* wavelet_data_v);


wavelet_coeff_t* wavelet_coeff_alloc( int k, int n);

void wavelet_coeff_release( wavelet_coeff_t* wavelet_coeff_v);

#endif