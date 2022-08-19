/*
 *	Author: Cheng Long
 *	Email: clong@cse.ust.hk
 */



#include "bst.h"


bst_t* bst_ini( )
{
	bst_t* T;

	T = ( bst_t*)malloc( sizeof( bst_t));
	memset( T, 0, sizeof( bst_t));

	//Problem specific.
	T->max = - FLT_MAX;
	T->min = FLT_MAX;

	//s
	T->m_size += sizeof( bst_t);

	emp_stat_v.memory_v += sizeof( bst_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	return T;
}

void bst_release_sub( bst_node_t* x)
{
	if( x->left != NULL)
		bst_release_sub( x->left);
	if( x->right != NULL)
		bst_release_sub( x->right);

	free( x);

	//s
	emp_stat_v.memory_v -= sizeof( bst_node_t);
	//s
}

/*
 *	Release the binary search tree T.
 */
void bst_release( bst_t* T)
{
	bst_release_sub( T->root);
	free( T);

	//s
	emp_stat_v.memory_v -= sizeof( bst_t);
	//s
}

void bst_insert( bst_t* T, bst_node_t* z)
{
	bst_node_t* y, *x;

	y = NULL;
	x = T->root;

	while( x != NULL)
	{
		y = x;
		if( z->key < x->key)
			x = x->left;
		else
			x = x->right;
	}

	z->p = y;
	if( y == NULL)
		T->root = z;	//T was empty.
	else if( z->key < y->key)
		y->left = z;	
	else
		y->right = z;

	//Problem specific.
	T->node_n ++;
	if( T->max < z->key)
		T->max = z->key;
	if( T->min > z->key)
		T->min = z->key;

	//s
	T->m_size += sizeof( bst_node_t);
	//s
}

/*
 *
 */
bst_node_t* bst_insert_key( bst_t* bst_v, KEY_TYPE key)
{
	bst_node_t* bst_node_v;

	bst_node_v = ( bst_node_t*)malloc( sizeof( bst_node_t));
	memset( bst_node_v, 0, sizeof( bst_node_t));

	bst_node_v->key = key;
	bst_insert( bst_v, bst_node_v);

	//s
	bst_v->m_size += sizeof( bst_node_t);

	emp_stat_v.memory_v += sizeof( bst_node_t);
	if( emp_stat_v.memory_v > emp_stat_v.memory_max)
		emp_stat_v.memory_max = emp_stat_v.memory_v;
	//s

	return bst_node_v;
}

void bst_transplant( bst_t* T, bst_node_t* u, bst_node_t* v)
{
	if( u->p == NULL)
		T->root = v;
	else if( u == u->p->left)
		u->p->left = v;
	else
		u->p->right = v;
	if( v != NULL)
		v->p = u->p;
}


bst_node_t* bst_min( bst_node_t* x)
{
	while( x->left != NULL)
		x = x->left;
	return x;
}

bst_node_t* bst_max( bst_node_t* x)
{
	while( x->right != NULL)
		x = x->right;
	return x;
}


void bst_delete( bst_t* T, bst_node_t* z)
{
	bst_node_t* y;

	if( z->left == NULL)
		bst_transplant( T, z, z->right);
	else if( z->right == NULL)
		bst_transplant( T, z, z->left);
	else
	{
		y = bst_min( z->right);
		if( y->p != z)
		{
			bst_transplant( T, y, y->right);
			y->right = z->right;
			y->right->p = y;
		}

		bst_transplant( T, z, y);
		y->left = z->left;
		y->left->p = y;
	}

	//Problem specific.
	T->node_n --;

	//s
	T->m_size -= sizeof( bst_node_t);
	//s
}

void in_order_walk( bst_node_t* x)
{
	if( x != NULL)
	{
		in_order_walk( x->left);
		printf( "%.0f  ", x->key);
		in_order_walk( x->right);
	}
}

/*
 *	
 *	@return null if no successor exists.
 */
bst_node_t*  bst_successor( bst_node_t* x)
{
	bst_node_t* y;

	if( x->right != NULL)
		return bst_min( x->right);

	y = x->p;
	while( y != NULL && x == y->right)
	{
		x = y;
		y = y->p;
	}

	return y;
}

/*
 * @return null if no predecessor exists.
 */
bst_node_t* bst_predecessor( bst_node_t* x)
{
	bst_node_t* y;

	if( x->left != NULL)
		return bst_max( x->left);

	y = x->p;
	while( y != NULL && x == y->left)
	{
		x = y;
		y = y->p;
	}

	return y;
}

void test_bst( )
{
	int i, node_n;
	bst_t* T;
	bst_node_t* x;

	T = bst_ini( );

	printf( "Testing the binary search tree...\n");
	printf( "Please input the number of nodes: ");
	scanf( "%i", &node_n);

	//srand( time( NULL));
	for( i=0; i<node_n; i++)
	{
		x = ( bst_node_t*)malloc( sizeof( bst_node_t));
		memset( x, 0, sizeof( bst_node_t));

		x->key = float( rand( )%100);
		bst_insert( T, x);
	}

	printf( "Tree constructed...\n");
	printf( "In-order walking...\n");
	in_order_walk( T->root);
	printf( "\n");

	printf( "\nDelete the root...\n");
	bst_delete( T, T->root);

	printf( "In-order walking...\n");
	in_order_walk( T->root);
	printf( "\n");

	printf( "Retrieving the successor of the root...\n");
	x = bst_successor( T->root);
	printf( "The key of the root is: %.0f\n", T->root->key);
	if( x != NULL)
		printf( "The key of the successor of the root is: %.0f\n", x->key);
	else
		printf( "No predecessors!\n");

	printf( "Retrieving the predecessor of the root...\n");
	x = bst_predecessor( T->root);
	printf( "The key of the root is: %.0f\n", T->root->key);
	if( x != NULL)
		printf( "The key of the predecessor of the root is: %.0f\n", x->key);
	else
		printf( "No successors!\n");

	printf( "Testing finished!\n");
	
	bst_release( T);
}