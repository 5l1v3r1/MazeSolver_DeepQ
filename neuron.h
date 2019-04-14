/*
 *	Neuron.h
 *
 *	Created: 13.2.2019 23:41:12
 * 	Author: Martin Horvath
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <iostream>
#include <time.h>
#include <math.h>

namespace AI 
{
	/********************************************************/
	/*      Struktura stavebneho prvku siete - neuronu      */
	/********************************************************/
	struct neuron
	{
		/* data */
		float	 *input;	// vstupy neuronu
		int		      N;    // pocet vstupov z okoliteho sveta
		float* dWeights;    // delta vahy neuronu (netreba)
		float*  Weights;    // vahy neuronu
		float	    out;    // vystup
		float     sigma;	// chyba neuronu (fitness)
		float	  alpha;    // learning rate (netreba)
		float  momentum;	// koeficient hybnosti (netreba)
		/* address */
		int num_of_next_neurons;	// for Backpropagate (netreba)
		int num_of_prev_neurons;
		struct neuron** next_neurons; // for Backpropagate (netreba)
		struct neuron** prev_neurons; // list_of_connected_neurons
	};
	typedef struct neuron* Neuron;

	/********************************************************/
	/*                   Definicie funkcii                  */
	/********************************************************/
	Neuron create_neuron(int _N);
	void add_connection_to_neuron(Neuron _n1, Neuron _n2);
	void activate_neuron(Neuron _n);
	void training_neuron(Neuron _n);
}

#endif /* NEURON_H_ */