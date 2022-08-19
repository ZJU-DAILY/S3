/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */


#include "RN_appro.h"
#include "min_error.h"

stat_t stat_v;
emp_stat_t emp_stat_v;

LOC_TYPE precision_thr_v2;

int	MIN_HEAP_OPT;

int main( )
{
	
	//real_data_preprocess( );

	//test2( );

	//test_bst( );

	//test_RN_simplify( );

	//collect_non_passing_stat( );

#ifndef WIN32
	//combine_files( 2);

	//prepare_cluster_data_v1( );

	//compress_data_set( );

	//compress_data_set_v2( );

	//collect_statistics( );
#endif

	//collect_statistics_clustering_datasets( );

	//printf( "%lf\n", pow( 0, 2));

	//emp_DPTS( );

	emp_DPTS_v2( );

	//test_HWT( );
	
	//emp_cats( );

	//prepare_cluster_data_v2( );

	//collect_clustering_measures( );

	return 0;
}
