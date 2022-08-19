
/*
 *	wavelet.h includes the APIs for Haar Wavelet Transformation.
 */

#ifndef WAVELET_H
#define WAVELET_H

#include "data_struct.h"
#include "data_utility.h"
#include "b_heap.h"


wavelet_coeff_t* HWT( wavelet_data_t* wavelet_data_v, int k, int opt);

wavelet_coeff_t* HWT_bottom_up( wavelet_data_t* wavelet_data_v, int k);

wavelet_coeff_t* HWT_basis_vector( wavelet_data_t* wavelet_data_v, int k);

wavelet_coeff_t* get_top_k( LOC_TYPE* values, int size, int k);

wavelet_data_t* HWDT( wavelet_coeff_t* wavelet_coeff_v);

//Testing APIs;
void test_HWT( );

wavelet_config_t* read_wavelet_config( );

wavelet_data_t* read_wavelet_data( wavelet_config_t* config_v);

void print_wavelet_data( wavelet_data_t* wavelet_data_v, char* f_name);

void print_wavelet_coeff( wavelet_coeff_t* wavelet_coeff_v, char* f_name);

#endif