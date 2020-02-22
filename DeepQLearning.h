#pragma once

/*
 *	DeepQLearning.h
 *
 *	Created: 13.2.2019 23:41:12
 * 	Author: Martin Horvath
 */

#ifndef DEEPQLEARNING_H_
#define DEEPQLEARNING_H_

#include <iostream>
#include <math.h>

#include "DeepNetwork.h"

namespace AI
{
	typedef struct
	{
		/* data */
		DeepNetwork* netQ;
		DeepNetwork* netQTarget;
		float *state, *newState;
		float reward;
		bool done;
		int action;
		float epsilon;
		float gamma;
	} DeepQNetwork;
	

	/********************************************************/
	/*                   Definicie funkcii                  */
	/********************************************************/
	DeepQNetwork* create_Qnetwork(int _num_of_layers, int* _num_of_neurons);
	int do_action(DeepNetwork* _net, float* _state, float _epsilon);
	void training_Qnetwork(DeepQNetwork* _net);
}

#endif /* DEEPQLEARNING_H_ */
