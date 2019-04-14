#pragma once

/*
 *	DeepLearning.h
 *
 *	Created: 13.2.2019 23:41:12
 * 	Author: Martin Horvath
 */

#ifndef DEEPNETWORK_H_
#define DEEPNETWORK_H_

#include <iostream>
#include <math.h>

#include "neuron.h"

namespace AI
{
	/********************************************************/
	/*      Struktura stavebneho prvku siete - neuronu      */
	/********************************************************/
	typedef struct {
		/* data */
		int num_of_layers;
		int* num_of_neurons;
		Neuron** net;
	} DeepNetwork;
	
	/********************************************************/
	/*                   Definicie funkcii                  */
	/********************************************************/
	DeepNetwork* create_network(int _num_of_layers, int* _num_of_neurons);
	void activate_network(DeepNetwork* _net, float* _input);
	float get_output_from_network(DeepNetwork* _net, int _index);
	void get_error_of_network(DeepNetwork* _net, float* _target);
	void training_network(DeepNetwork* _net, float* _target);
	float tanh_derivation(float y);
}

#endif /* DEEPNETWORK_H_ */