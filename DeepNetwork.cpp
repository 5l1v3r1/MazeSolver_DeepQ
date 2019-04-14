#include "DeepNetwork.h"

namespace AI
{
	/**************************************/
	/*            Vytvor siet             */
	/**************************************/
	DeepNetwork* create_network(int _num_of_layers, int* _num_of_neurons)
	{
		DeepNetwork* _net = (DeepNetwork*)malloc(sizeof(*_net));

		_net->num_of_layers = _num_of_layers;
		_net->num_of_neurons = _num_of_neurons;
		_net->net = (Neuron**)malloc(_num_of_layers * sizeof(Neuron*));
		for (int i = 0; i < _num_of_layers; i++)
		{
			_net->net[i] = (Neuron*)malloc(_num_of_neurons[i] * sizeof(Neuron));
			for (int j = 0; j < _num_of_neurons[i]; j++)
			{
				if (i == 0)
					_net->net[i][j] = create_neuron(_num_of_neurons[0]);
				else
				{
					_net->net[i][j] = create_neuron(_num_of_neurons[i-1]);

					for (int k = 0; k < _num_of_neurons[i-1]; k++)
						add_connection_to_neuron(_net->net[i][j], _net->net[i-1][k]);
				}
			}
		}

		return _net;
	}

	/**************************************/
	/*         Vyjadri chybu siete        */
	/**************************************/
	void get_error_of_network(DeepNetwork* _net, float* _target)
	{
		for (int j = 0; j < _net->num_of_neurons[_net->num_of_layers - 1]; j++)
			_net->net[_net->num_of_layers - 1][j]->sigma = (_target[j] - _net->net[_net->num_of_layers - 1][j]->out) * tanh_derivation(_net->net[_net->num_of_layers - 1][j]->out);
				
		for (int i = _net->num_of_layers - 2; i >= 0; i--)
		{
			for (int j = 0; j < _net->num_of_neurons[i]; j++)
			{
				_net->net[i][j]->sigma = 0.0f;
				for (int k = 0; k < _net->net[i][j]->num_of_next_neurons; k++)
				{
					_net->net[i][j]->sigma += _net->net[i][j]->next_neurons[k]->sigma * _net->net[i][j]->next_neurons[k]->Weights[j];
				}
				_net->net[i][j]->sigma *= tanh_derivation(_net->net[i][j]->out);
			}
		}
	}

	/**************************************/
	/*             Aktivuj siet           */
	/**************************************/
	void activate_network(DeepNetwork* _net, float* _input)
	{
		for (int j = 0; j < _net->num_of_neurons[0]; j++)
		{
			_net->net[0][j]->input = _input;
		}

		for (int i = 0; i < _net->num_of_layers; i++)
		{
			for (int j = 0; j < _net->num_of_neurons[i]; j++)
			{
				activate_neuron(_net->net[i][j]);
			}
		}
	}

	/**************************************/
	/*           Pretrenuj siet           */
	/**************************************/
	void training_network(DeepNetwork* _net, float* _target)
	{
		get_error_of_network(_net, _target);

		for (int i = 0; i < _net->num_of_layers; i++)
		{
			for (int j = 0; j < _net->num_of_neurons[i]; j++)
			{
				training_neuron(_net->net[i][j]);
			}
		}
	}

	float get_output_from_network(DeepNetwork* _net, int _index)
	{
		return _net->net[_net->num_of_layers - 1][_index]->out;
	}

	float tanh_derivation(float y)
	{
		return (1.0f - (y * y));
	}
}