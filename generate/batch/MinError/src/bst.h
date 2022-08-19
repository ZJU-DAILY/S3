/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */


/*
 *	1. The implementation of the binary search tree follows pseudo-code in the textbook "Introduction to Algorithms".
 */

#ifndef BST_H
#define	BST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

#include "data_utility.h"

#define	KEY_TYPE	float

extern emp_stat_t	emp_stat_v;

/*The structure of the node in bst.*/
typedef struct bst_node
{
	KEY_TYPE	key;

	//For Span-Search
	int			range_inx;

	struct bst_node*	p;
	struct bst_node*	left;
	struct bst_node*	right;

}	bst_node_t;

/*The structure of a bst.*/
typedef struct bst
{
	bst_node_t* root;
	
	//Problem specific information.
	int			node_n;

	KEY_TYPE	min;
	KEY_TYPE	max;

	//For empirical study.
	float		m_size;

	//For span-search
	//bst_node_t*	min_node;
	//bst_node_t* max_node;

}	bst_t;


bst_t* bst_ini( );

void bst_release_sub( bst_node_t* x);

void bst_release( bst_t* T);

void bst_insert( bst_t* T, bst_node_t* z);

bst_node_t* bst_insert_key( bst_t* bst_v, KEY_TYPE key);

void bst_transplant( bst_t* T, bst_node_t* u, bst_node_t* v);

bst_node_t* bst_min( bst_node_t* x);

bst_node_t* bst_max( bst_node_t* x);

void bst_delete( bst_t* T, bst_node_t* z);

void in_order_walk( bst_node_t* x);

bst_node_t* bst_successor( bst_node_t* x);

bst_node_t* bst_predecessor( bst_node_t* x);

void test_bst( );


#endif