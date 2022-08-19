/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



#ifndef	DATA_STRUCT_H
#define	DATA_STRUCT_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>


#define		LOC_TYPE	float
#define		DIST_TYPE	float


//The structure for storing the location information.
typedef struct loc
{
	LOC_TYPE	x;
	LOC_TYPE	y;

}	loc_t; 

//The structure for storing the road network [adjacency list].
typedef struct RN_graph_list
{
	int			id;			
	//loc_t		loc_v;	
	
	DIST_TYPE	leng;		//the lenghth of the road segment.		
	//float		slope;		//the "absolute" slope of the road segment.

	struct RN_graph_list*	next;

	//Problem specific variables.
	int			tag;		//used for RN_decompose.

}	RN_graph_list_t;

typedef struct RN_graph_head
{
	loc_t				loc_v;
	int					degree;

	RN_graph_list_t*	list;

}	RN_graph_head_t;

typedef struct RN_graph
{
	int					vertex_n;
	int					edge_n;

	RN_graph_head_t*	head_v;		//Pointer array.

	//For RN_simplify.
	int					vertex_n_sim;

	//For empirical study.
	float				m_size;

}	RN_graph_t;




//The structure for storing a road.
typedef struct R_node
{
	int			id1;
	int			id2;

	loc_t*		loc_v1;			//Redundancy information for the sake of bidirectional list.
	loc_t*		loc_v2;
	
	DIST_TYPE	leng;
	float		slope;

	float		t_interval;

	struct R_node*	next;

	//
	int			slope_tag;

}	R_node_t;

typedef struct R_list
{
	int				size;		//The number of segments in the trajectory.

	R_node_t*		head;
	R_node_t*		rear;		//

	struct R_list*	next;		//for R_set.

	//For Empirical study.
	float			m_size;

}	R_list_t;


//The structure for storing the set of roads (components).
typedef struct R_set
{
	int			num;

	R_list_t*	head;	//The roads are linked via list.

	//For empirical study.
	float		m_size;

}	R_set_t;




//The structure for storing the graph for the SP-Method algorithm.
typedef struct eps_g_node
{
	int		id;

	struct	eps_g_node* next;

	//For RN_simplify.
	int		ori_id;		//The intuition is that the id's might not in the order of 0, 1, 2, ....
	
}	eps_g_node_t;

typedef struct eps_g_head
{
	eps_g_node_t*	list;
	
	//For RN_simplify.
	int		ori_id;		//The intuition is that the id's might not in the order of 0, 1, 2, ....

}	eps_g_head_t;

typedef struct eps_g
{
	int				vertex_n;
	int				edge_n;
	eps_g_head_t*	head;

	//For empirical study.
	float	m_size;

}	eps_g_t;



//The structure for storing the slope range information.
typedef struct slope_r
{
	float	s_sta;
	float	s_end;

}	slope_r_t;

typedef struct slope_r_node
{
	slope_r_t				slope_r_v;		//Consider using non-pointer.
	struct slope_r_node*	next;

}	slope_r_node_t;

typedef struct slope_r_list
{
	int						num;
	slope_r_node_t*			rear;			//For adding entries.
	slope_r_node_t*			head;

}	slope_r_list_t;


//The structure for the search space information.
typedef struct search_space
{
	float*	s_space;
	int		size;

}	search_space_t;


//----------------------Modified on 26, Feb 2012-----------------------

//The structure for storing the span information.
typedef struct span_m
{
	float	min_1;		//[0, \pi)
	float	max_1;

	float	min_2;		//[\pi, 2\pi)
	float	max_2;

}	span_m_t;


//----------------------Modified on 26, Feb 2012-----------------------

//The structure for storing the raw trajectory data.
typedef struct	triplet
{
	loc_t	loc_v;

	int		year;
	int		month;
	int		day;
	int		hour;
	int		min;
	int		sec;

}	triplet_t;

typedef	struct tra_node
{
	int					pos_id;		//starting from 0.
	triplet_t			tri_v;
	struct tra_node*	next;

}	tra_node_t;

typedef struct tra_list
{
	int				pos_n;
	tra_node_t*		head;
	tra_node_t*		rear;

}	tra_list_t;


//The structure for storing a sequence of numbers.
typedef struct id_node
{
	int					pos_id;
	struct id_node*		next;

}	id_node_t;

typedef struct id_list
{
	int			pos_id_n;
	
	id_node_t*	head;
	id_node_t*	rear;

}	id_list_t;


//The structure for supporting the implementation of SP-Method-E2.
typedef struct hop_node
{
	int					pos_id;
	//int					pre_id;
	
	R_node_t*			seg_before;		//The segment ends at the position with id.
	R_node_t*			seg_after;		//The segment starts at the position with id.

	struct hop_node*	next;
	struct hop_node*	pre;

}	hop_node_t;

typedef struct hop_list
{
	int					hop_order;
	hop_node_t*			head;
	hop_node_t*			rear;

	struct hop_list*	next;

}	hop_list_t;


//The structure for maintaining the fdr's incrementally.
typedef struct fdr_array
{
	int				seg_n;		
	int*			rear_i;	//For maintainence on-demand only.
							//Indicate the last maintained entry.
		
	slope_r_list***	fdr;

}	fdr_array_t;


//The alternative structure (array) for the R_list_t struct
typedef struct R_array
{
	int			size;
	R_node_t**	array;
	
}	R_array_t;


//The structure for implementing the Merge algorithm.
typedef struct group_node
{
	//int				g_id;

	int					sta_seg_id;
	int					end_seg_id;

	struct group_node*	next;
	struct group_node*	pre;

	//For Merge_sub2 only.
	int					loc;

}	group_node_t;


typedef struct group_list
{
	int				size;
	
	group_node_t*	head;
	group_node_t*	rear;

}	group_list_t;


//The structure for implementing the Dougolas-Peucker algorihtm.
typedef struct part_node
{
	//int			pos_id_sta;
	//int			pos_id_end;

	R_node_t*		sta;
	R_node_t*		end;
	R_node_t*		split_loc;		//The splitting point is the second end point.

	//int			pos_id_split;
	double			largest_dist;

	struct part_node*	next;

}	part_node_t;

typedef struct part_list
{
	int				size;

	part_node_t*	head;
	part_node_t*	rear;

}	part_list_t;


//For measuring the clustering quality.
typedef struct seg
{
	loc_t loc_v1;
	loc_t loc_v2;

}	seg_t;

typedef struct cluster_measure
{
	int	TP;
	int FP;
	int TN;
	int	FN;

}	cluster_measure_t;


//Starting from 17 July, 2013.
//For the Min-Error problem.

//The structure for storing the error space.
typedef struct error_space
{
	float**	matrix;		//matirx[i][j] is the error of segment p_ip_j.
	int		pos_n;

}	error_space_t;


//Structures for the storing the span space.
typedef struct array
{
	int		s;
	int		e;
	int		j;

}	array_t;

typedef struct array_node
{
	array_t				array_v;

	struct array_node	*pre;
	struct array_node	*next;

}	array_node_t;

typedef struct span_space
{
	int				array_n;
	array_node_t*	head;

#ifndef WIN32
	long long		size;		//the size of the search space.
#else
	int			size;
#endif

	int				pos_n;
	float*			slope_array;

}	span_space_t;

//Structure of storing a angular range.
typedef struct angular_range
{
	float	theta_1;
	float	theta_2;

}	angular_range_t;


//Starting from April 17, 2014
//typedef struct trace_data
//{
//	int			size;
//	LOC_TYPE*	x_array;
//	LOC_TYPE*	y_array;

//}	trace_data_t;

//For the haar wavelet transform (HWT)
typedef struct wavelet_coeff
{
	int			k;		//the number of coefficients
	LOC_TYPE*	coeff;	//the coefficient array
	int*		idx;	//the indexes of the coefficents.

	int			n;

	//int		m;		//indicating the original # of elements is 2^m
	//int		n;		//indicating that the original # of elements is n

}	wavelet_coeff_t;


typedef struct wavelet_data
{
	int			n;		//the # of elements
	LOC_TYPE*	value;	//the array storing the elements

	int			m;		//m = log n, i.e., n = 2^m

}	wavelet_data_t;


#endif
