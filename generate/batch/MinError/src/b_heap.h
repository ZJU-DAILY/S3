/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



/*
*	1. The mechanism of "cmp" function and the "void*" type involved there.
*	2. The choice min-heap or max-heap can be determined by changing the function of "cmp".
*	3. The heap array's subscript starts from zero.
*	4. The concept of "heap array" and "object array", which work collaborarily.
*
*	5. [Updated on 17 Feb. 2012]
*		(a). Variable MIN_HEAP_OPT is variable for specifying min-heap or max-heap.
*		(b). Object type should maintain an element 'key' for the heap structure.
*/


#ifndef B_HEAP_H
#define	B_HEAP_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include "data_struct.h"
#include "data_utility.h"


#define		B_CONFIG_FILE		"b_config.txt"
#define		MAX_FILENAME_LENG	256
#define		FLOAT				float

//#define		MIN_HEAP_OPT		0

extern		int	MIN_HEAP_OPT;

/*
#define		B_INPUT_FILE	"b_input.txt"
#define		B_OUTPUT_FILE	"b_output.txt"
*/

extern emp_stat_t	emp_stat_v;

//The structure of the object that indexed by the binary heap.
typedef	struct h_obj
{
	int		id;
	int		loc;				

	//object-specific variables.
	float	key;	//The key of the heap. 

	//For the Merge algorithm.
	group_node_t* group_node_v;

	//For the Dogoulas-Peucker algorithm.
	part_node_t* part_node_v;
	
	int		pre;
	int		next;

	//For Merge-Error.
	R_node_t*	R_node_v_sta;
	R_node_t*	R_node_v_end;
//	R_node_t*	R_node_v_sta2;
//	R_node_t*	R_node_v_end2;

	//For Merge-Span.
	span_m_t	span_m_v;
		
	//For Merge-Span2 (only)
	R_node_t*	R_node_v_sta1;
	R_node_t*	R_node_v_end1;

	//For Span-Search.
	

} h_obj_t;


//The structure of the node of a heap.
typedef	struct b_heap
{
	int			size;			//the heap's size.
	int			rear;			//the heap's rear tag.
	h_obj_t*	obj_arr;		//the object array.
	int*		h_arr;			//the heap array.

	//For empirical study.
	float		m_size;

} b_heap_t;



//-------------------------------------

//int cmp( const void* obj_v, int fir, int sec);

int cmp_min( int* array, int n1, int n2, h_obj_t* obj_v);

int cmp_max( int* array, int n1, int n2, h_obj_t* obj_v);

//-------------------------------------

b_heap_t* alloc_b_heap( int size);

void release_b_heap( b_heap_t* b_h);

void b_t_heapify( int* array, int cur, h_obj_t* obj_v);

void t_b_heapify( int* array, int cur, int rear, h_obj_t* obj_v);

int insert_b_heap( b_heap_t* b_h, int n);

int get_top( b_heap_t* b_h);

void update_key( b_heap_t* b_h, int k);

//-------------------------------------


void print_b_heap( FILE* o_fp, b_heap_t* b_h);

void test_b_heap( );


#endif