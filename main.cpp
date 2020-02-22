// MazeSolver_v3.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include <thread>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <windows.h>

// AI
#include "DeepNetwork.h"
#include "DeepQLearning.h"

#define POCET_KROKOV						      (40)
#define POCET_ITERACII						 (1000000)

using namespace AI;
using namespace std;

	const int AreaWidth = 10;
	const int AreaHeight = 10;
	int* Environment = new int[AreaWidth * AreaHeight];

	enum Objects
	{
		Nothing = 0,
		Line,
		End
	};

	enum Actions
	{
		Up = 0,
		Down,
		Left,
		Right
	};

	struct Player
	{
		int positionX;
		int positionY;
	};


	void NakresliHernySvet(Player _player);
	int upravaDatSenzorov(float* _state, Player _player);

	int main(int argc, char** argv)
	{
		Player robot;
		FILE *f_tab, *f_err;
		float error_avg;

		f_tab = fopen("tab.txt", "w");
		f_err = fopen("err.txt", "w");

		//  AI
		DeepQNetwork* _ai = create_Qnetwork(3, new int[3]{ 7, 32, 4 }); 
		int penalties = 0, pom;

		// init random
		srand((unsigned)time(NULL));
 
		// Clear environment
		for (int i = 0; i < (AreaWidth * AreaHeight); i++)
			Environment[i] = (int)Objects::Nothing;

		// Set lines
		Environment[2 + 2 * AreaWidth] = (int)Objects::Line;
		Environment[3 + 2 * AreaWidth] = (int)Objects::Line;
		Environment[4 + 2 * AreaWidth] = (int)Objects::Line;
		Environment[5 + 2 * AreaWidth] = (int)Objects::Line;
		Environment[6 + 2 * AreaWidth] = (int)Objects::Line;
		Environment[7 + 2 * AreaWidth] = (int)Objects::Line;
		Environment[4 + 3 * AreaWidth] = (int)Objects::Line;
		Environment[4 + 4 * AreaWidth] = (int)Objects::Line;
		Environment[4 + 5 * AreaWidth] = (int)Objects::Line;
		Environment[4 + 6 * AreaWidth] = (int)Objects::Line;
		Environment[4 + 7 * AreaWidth] = (int)Objects::Line;
		Environment[3 + 5 * AreaWidth] = (int)Objects::Line;
		Environment[2 + 5 * AreaWidth] = (int)Objects::Line;
		Environment[2 + 6 * AreaWidth] = (int)Objects::Line;
		Environment[7 + 3 * AreaWidth] = (int)Objects::Line;
		Environment[7 + 4 * AreaWidth] = (int)Objects::Line;

		// Set end
		Environment[5 + 7 * AreaWidth] = (int)Objects::End;

		/* Training */
		for (int time = 0, step; time <= POCET_ITERACII; time++)
		{
			// nove kolo hry
			_ai->done = false;
			penalties = 0;
			error_avg = 0;

			// zaciatocna poloha hraca
			robot.positionX = 2;
			robot.positionY = 2;

			// pociatocny state
			upravaDatSenzorov(_ai->state, robot);

			for (step = 0; step < POCET_KROKOV; step++)
			{
				// Vyber akciu
				_ai->action = do_action(_ai->netQ, _ai->state, _ai->epsilon);

				// Vykonaj akciu
				switch (_ai->action)
				{
				case (int)Actions::Up:
					if (robot.positionY > 0) robot.positionY -= 1;
					break;
				case (int)Actions::Down:
					if (robot.positionY < (AreaHeight - 1)) robot.positionY += 1;
					break;
				case (int)Actions::Left:
					if (robot.positionX > 0) robot.positionX -= 1;
					break;
				case (int)Actions::Right:
					if (robot.positionX < (AreaWidth - 1)) robot.positionX += 1;
					break;
				}

				// newState
				pom = upravaDatSenzorov(_ai->newState, robot);

				// ziskaj reward
				if (Environment[pom] == (int)Objects::Nothing) {
					penalties++;  
					_ai->reward = -1.0f; 
					_ai->done = true; 
				}
				else if (Environment[pom] == (int)Objects::End) { 
					_ai->reward = +1.0f; 
					_ai->done = true;
				}
				else {
					_ai->reward = -0.01f; 
					_ai->done = false;
				}
				
				training_Qnetwork(_ai);

				if (time > 0 && time % 10000 == 0)
				{
					system("clear");

					NakresliHernySvet(robot);

					std::cout << "Training ... " << time / 10000.0 << "%" << std::endl;
					std::cout << "Penalties = " << penalties << std::endl;

					std::cout << "state = " << pom << std::endl;
					std::cout << "action = " << _ai->action << std::endl;
					std::cout << "sigma1 = " << _ai->netQ->net[_ai->netQ->num_of_layers - 1][0]->sigma << std::endl;
					std::cout << "sigma2 = " << _ai->netQ->net[_ai->netQ->num_of_layers - 1][1]->sigma << std::endl;
					std::cout << "sigma3 = " << _ai->netQ->net[_ai->netQ->num_of_layers - 1][2]->sigma << std::endl;
					std::cout << "sigma4 = " << _ai->netQ->net[_ai->netQ->num_of_layers - 1][3]->sigma << std::endl;					
				}

				// save error
				error_avg += (_ai->netQ->net[_ai->netQ->num_of_layers - 1][_ai->action]->sigma 
					* _ai->netQ->net[_ai->netQ->num_of_layers - 1][_ai->action]->sigma);

				for (int i = 0; i < 7; i++)
					_ai->state[i] = _ai->newState[i];

				if (_ai->done != 0) break;
			}

			// error avg
			error_avg = error_avg / step;
			error_avg = sqrtf(error_avg);

			if (time > 0 && time % 1000 == 0 && !isinf(error_avg))
			{
				std::cout << "error_avg = " << error_avg << std::endl;
				fprintf(f_err, "error_avg;%f\n", error_avg);
			}
		}

		// Pause
		system("pause");

        /*  Evaluate agent's performance after Q-learning  */
		penalties = 0;
		for (int time = 0; time < 5; time++)
		{
			// nove kolo hry
			_ai->done = false;

			// zaciatocna poloha
			robot.positionX = 2;
			robot.positionY = 2;

			// pociatocny state
			pom = upravaDatSenzorov(_ai->state, robot);

			this_thread::sleep_for(std::chrono::milliseconds(500));
			system("clear");

			NakresliHernySvet(robot);

			do {
				this_thread::sleep_for(std::chrono::milliseconds(500));
				system("clear");

				// Vyber akciu
				_ai->action = do_action(_ai->netQ, _ai->state, 0.0f);

				if (time == 0)
					fprintf(f_tab, "%d;%d;%d;%d;%d;%d;%d;;%d;%d\n", (int)_ai->state[0], (int)_ai->state[1], (int)_ai->state[2], (int)_ai->state[3], (int)_ai->state[4], (int)_ai->state[5], (int)_ai->state[6], (_ai->action >> 1) & 0x01, _ai->action & 0x01);

				// Vykonaj akciu
                switch (_ai->action)
                {
                        case (int)Actions::Up:
                            if (robot.positionY > 0) robot.positionY -= 1;
                            break;
                        case (int)Actions::Down:
                            if (robot.positionY < (AreaHeight-1)) robot.positionY += 1;
                            break;
                        case (int)Actions::Left:
                            if (robot.positionX > 0) robot.positionX -= 1;
                            break;
                        case (int)Actions::Right:
                            if (robot.positionX < (AreaWidth-1)) robot.positionX += 1;
                          break;
				}

				NakresliHernySvet(robot);

				std::cout << "state = " << pom << std::endl;
				std::cout << "action = " << _ai->action << std::endl;

				// state
				pom = upravaDatSenzorov(_ai->state, robot);

				// ziskaj reward
				if (Environment[pom] == (int)Objects::Nothing) { penalties++;  /*done = true;*/ }
				else if (Environment[pom] == (int)Objects::End) { _ai->done = true; }									
			} while (!_ai->done);
		}

		std::cout << "Penalties for 5 games = " << penalties << std::endl;

		fclose(f_tab);
		fclose(f_err);

		return 0;
	}

	void NakresliHernySvet(Player _player)
	{
		for (int j = 0; j < AreaHeight; j++)
		{
			std::cout << "|";
			for (int i = 0; i < AreaWidth; i++)
			{
				if (_player.positionX == i && _player.positionY == j)
				{
				    //SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (BACKGROUND_RED|FOREGROUND_INTENSITY));					
					std::cout << "\033[0;41m O \033[0m";
				    //SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (0|FOREGROUND_INTENSITY));
				}
				else
				{
					switch (Environment[(j * AreaWidth) + i])
					{
					case (int)Objects::Nothing:
						//Console.ResetColor();
						std::cout << "\033[0m   ";
						break;
					case (int)Objects::Line:
						//Console.BackgroundColor = ConsoleColor.White;
						std::cout << "\033[0;47;30m X \033[0m";
						//Console.ResetColor();
						break;
					case (int)Objects::End:
					    //SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (BACKGROUND_GREEN));
						//Console.BackgroundColor = ConsoleColor.Green;
						//Console.ForegroundColor = ConsoleColor.Black;
						std::cout << "\033[0;42;30m E \033[0m";
					    //SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (0|FOREGROUND_INTENSITY));
						//Console.ResetColor();
						break;
					}
				}
			}
			std::cout << "|" << std::endl;
		}
		std::cout << std::endl;
	}

	int upravaDatSenzorov(float* _state, Player _player)
	{
		int pom = (_player.positionY * AreaWidth) + _player.positionX;

		for (int i = 0; i < 7; i++)
		{
			_state[i] = (float) ((pom >> i) & 0x01);
		}

		return pom;
	}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
