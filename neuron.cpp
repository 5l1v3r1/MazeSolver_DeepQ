#include "neuron.h"

namespace AI
{
	/**************************************/
	/*           Vytvor neuron            */
	/**************************************/
	Neuron create_neuron(int _N)
	{
		Neuron _neuron = (Neuron)std::malloc(sizeof(*_neuron));

		srand((unsigned)time(NULL));

		_neuron->N = _N;
		_neuron->input = NULL;
		_neuron->dWeights = (float*)std::malloc((_N+1)*sizeof(float));	// +bias
		_neuron->Weights = (float*)std::malloc((_N+1) * sizeof(float));	// +bias
		
		//float k = 2.4f / (_N+1);
		for (int i = 0; i < _N+1; i++)
		{
			_neuron->dWeights[i] = 0.0f;
			_neuron->Weights[i] = (((float)rand()/(float)RAND_MAX) - 0.5f) * 2.0f;//(((float)rand()/(float)RAND_MAX) * (k + k)) - k;
		}
		
		_neuron->out = 0.0f;
		_neuron->sigma = 0.0f;
		_neuron->alpha = 0.1f;
		_neuron->momentum = 0.1f;
		_neuron->num_of_next_neurons = 0;
		_neuron->num_of_prev_neurons = 0;
		_neuron->prev_neurons = NULL;
		_neuron->next_neurons = NULL;

		return _neuron;
	}

	/*******************************************************/
	/*       Spoj vstup neuronu1 s vystupom neuronu2       */
	/*******************************************************/
	void add_connection_to_neuron(Neuron _n1, Neuron _n2)
	{
		if (!_n1->prev_neurons)
		{
			_n1->num_of_prev_neurons = 1;
			_n1->prev_neurons = (Neuron *)std::malloc(_n1->num_of_prev_neurons * sizeof(*_n1->prev_neurons));
			if (!_n1->prev_neurons) { _n1->num_of_prev_neurons = 0; return; }
			_n1->prev_neurons[0] = _n2;
		}
		else
		{
			_n1->num_of_prev_neurons++;
			_n1->prev_neurons = (Neuron *)std::realloc(_n1->prev_neurons, (_n1->num_of_prev_neurons * sizeof(*_n1->prev_neurons)));
			if (!_n1->prev_neurons) { _n1->num_of_prev_neurons = 0; return; }
			_n1->prev_neurons[_n1->num_of_prev_neurons - 1] = _n2;
		}

		if (!_n2->next_neurons)
		{
			_n2->num_of_next_neurons = 1;
			_n2->next_neurons = (Neuron *)std::malloc(_n2->num_of_next_neurons * sizeof(*_n2->next_neurons));
			if (!_n2->next_neurons) { _n2->num_of_next_neurons = 0; return; }
			_n2->next_neurons[0] = _n1;
		}
		else
		{
			_n2->num_of_next_neurons++;
			_n2->next_neurons = (Neuron *)std::realloc(_n2->next_neurons, (_n2->num_of_next_neurons * sizeof(*_n2->next_neurons)));
			if (!_n2->next_neurons) { _n2->num_of_next_neurons = 0; return; }
			_n2->next_neurons[_n2->num_of_next_neurons - 1] = _n1;
		}
	}

	/**************************************/
	/*           Aktivuj neuron           */
	/**************************************/
	void activate_neuron(Neuron _n)
	{
		_n->out = 0.0f;

		// Calc input from world
		if (_n->input != NULL)
		{
			for (int i = 0; i < _n->N; i++)
			{
				_n->out += _n->Weights[i] * _n->input[i];
			}
		}
		// Calc input from other neurons
		else if (_n->prev_neurons != NULL)
		{
			for (int i = 0; i < _n->N; i++)
			{
				_n->out += _n->Weights[i] * _n->prev_neurons[i]->out;
			}
		}

		_n->out += _n->Weights[_n->N]; // *1  (bias)
		_n->out = tanhf(_n->out);
	}

	/**************************************/
	/*          Pretrenuj neuron          */
	/**************************************/
	void training_neuron(Neuron _n)
	{
		// Calc with input from world
		if (_n->input != NULL)
		{
			for (int i = 0; i < _n->N; i++)
			{
				_n->dWeights[i] = (_n->alpha * _n->sigma * _n->input[i]) + (_n->momentum * _n->dWeights[i]);
				_n->Weights[i] += _n->dWeights[i];
			}
		}
		// Calc with input from other neurons
		else if (_n->prev_neurons != NULL)
		{
			for (int i = 0; i < _n->N; i++)
			{
				_n->dWeights[i] = (_n->alpha * _n->sigma * _n->prev_neurons[i]->out) + (_n->momentum * _n->dWeights[i]);
				_n->Weights[i] += _n->dWeights[i];
			}
		}

		_n->dWeights[_n->N] = (_n->alpha * _n->sigma /* (*1) */) + (_n->momentum * _n->dWeights[_n->N]);
		_n->Weights[_n->N] += _n->dWeights[_n->N];
	}
}