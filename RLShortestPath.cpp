#include <iostream>
#include "RLShortestPath.h"

using namespace RL;

int main()
{
	const uint32_t ITERATIONS = 700;
	const size_t GRAPHS_MATRIX_SIZE = 9;
	const size_t REWARD_MATRIX_SIZE = 9;
	const size_t QLEARN_MATRIX_SIZE = 9;
	const uint32_t GOAL_NODE = 7;
	const double GAMMA = 0.8f;


	RLShortestPath rlShortestPath(GRAPHS_MATRIX_SIZE, REWARD_MATRIX_SIZE, QLEARN_MATRIX_SIZE, GOAL_NODE, GAMMA);

	ArrayEdge edges = {
		createEdge(0, 2),
		createEdge(0, 1),
		createEdge(0, 3),
		createEdge(2, 4),
		createEdge(5, 6),
		createEdge(7, 4),
		createEdge(0, 6),
		createEdge(5, 3),
		createEdge(3, 7),
		createEdge(0, 8)
	};

	rlShortestPath.addEdges(edges);
	rlShortestPath.printGraphNodes();

	rlShortestPath.learn(ITERATIONS);

	ArrayInt steps = rlShortestPath.solve();
	rlShortestPath.printArrayDebug(steps);

	return 0;
}
