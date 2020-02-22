#include "DeepQLearning.h"

namespace AI
{
	DeepQNetwork* create_Qnetwork(int _num_of_layers, int* _num_of_neurons)
	{
		DeepQNetwork* _net = (DeepQNetwork*)malloc(sizeof(DeepQNetwork));

		_net->netQ = create_network(_num_of_layers, _num_of_neurons);
		_net->netQTarget = create_network(_num_of_layers, _num_of_neurons);
		_net->state = new float[_num_of_neurons[0]];
		_net->newState = new float[_num_of_neurons[0]];
		_net->reward = 0;
		_net->action = 0;
		_net->epsilon = 0.15f;
		_net->gamma = 0.95f;

		return _net;
	}

	int do_action(DeepNetwork* _net, float* _state, float _epsilon)
	{
		int action;

		activate_network(_net, _state);
		if (((float)rand()/(float)RAND_MAX) < _epsilon)
		{
			action = rand() % _net->num_of_neurons[_net->num_of_layers-1];
		}
		else
		{
			action = 0;
			for (int i = 1; i < _net->num_of_neurons[_net->num_of_layers-1]; i++)
			{
				if (get_output_from_network(_net, i) > get_output_from_network(_net, action)) action = i;
			}
		}

		return action;
	}

	void training_Qnetwork(DeepQNetwork* _net)
	{
		float* qTarget = new float[_net->netQ->num_of_neurons[_net->netQ->num_of_layers-1]];

		// ziskaj QVal dalsieho stavu
		float newQVal = get_output_from_network(_net->netQTarget, do_action(_net->netQTarget, _net->newState, 0.0f));

		// priprav ocakavany vystup
		for (int i = 0; i < _net->netQ->num_of_neurons[_net->netQ->num_of_layers-1]; i++)
		{
			if (i == _net->action)
			{
				if (!_net->done)
					qTarget[i] = (_net->reward + _net->gamma * newQVal);
				else
					qTarget[i] = _net->reward;
			}
			else
				qTarget[i] = 0.0f;
		}
		training_network(_net->netQ, qTarget);

		// Copy W from Q to Qtarget network
		for (int i = 0; i < _net->netQ->num_of_layers; i++)
			for (int j = 0; j < _net->netQ->num_of_neurons[i]; j++)
				for (int k = 0; k < _net->netQ->net[i][j]->N+1; k++)
					_net->netQTarget->net[i][j]->Weights[k] = (0.01f)*_net->netQ->net[i][j]->Weights[k] + (1.0f-0.01f)*_net->netQTarget->net[i][j]->Weights[k];	
	}
}