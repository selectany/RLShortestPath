#pragma once
#ifndef __RLSHORTESTPATH__
#define __RLSHORTESTPATH__

#include <algorithm>
#include <numeric>
#include "RandomSelector.h"
#include "GlobalDefs.h"

namespace RL
{
	class RLShortestPath
	{
	public:
		explicit RLShortestPath(size_t nodesSize, size_t rewardsSize, size_t qlearnSize, uint32_t goalNode, double gamma) noexcept :
			GN(goalNode), GAMMA(gamma),
			matrixGraph(nodesSize, ArrayInt(nodesSize, 0)),
			matrixReward(rewardsSize, ArrayDouble(rewardsSize, -1.0f)),
			matrixQLearn(qlearnSize, ArrayDouble(qlearnSize, 0.0f)) {}

		void learn(const uint32_t iterations)
		{
			ArrayInt rndArr(matrixQLearn.size());
			std::iota(rndArr.begin(), rndArr.end(), 0); // initialize with range [0, qlearnMatrixSize - 1]

			for (size_t i = 0; i < iterations; ++i)
			{
				RandomSelector<> selector{};
				int currentState = selector(rndArr); // random [0, qlearnMatrixSize - 1]

				ArrayInt availableActions = getAvailableActions(currentState);
				int action = RLShortestPath::sampleNextAction(availableActions);

				update(currentState, action);
			}

			printQLearnMatrix();
		}

		ArrayInt solve() const
		{
			int currentState = 0;
			ArrayInt steps = { currentState };

			while (currentState != GN)
			{
				// Find next step
				const ArrayDouble& currentQLearnRow = matrixQLearn[currentState];

				int nextStepIndex;
				ArrayInt maxQLearnIndexes;
				const double maxQLearnVal = *(std::max_element(currentQLearnRow.begin(), currentQLearnRow.end()));
				for (size_t i = 0; i < currentQLearnRow.size(); ++i)
				{
					if (currentQLearnRow[i] == maxQLearnVal)
					{
						maxQLearnIndexes.push_back(i);
					}
				}

				if (maxQLearnIndexes.size() > 1)
				{
					RandomSelector<> selector{};
					nextStepIndex = selector(maxQLearnIndexes);
				}
				else
				{
					nextStepIndex = maxQLearnIndexes[0];
				}

				// add to result (step) array
				steps.push_back(nextStepIndex);

				// prepare for next step
				currentState = nextStepIndex;
			}

			return steps;
		}

		ArrayInt getAvailableActions(int state) const // state is a row of reward matrix i.e. it is reward node "From"
		{
			const ArrayDouble& currentStateRow = matrixReward[state];

			const size_t R = matrixReward.size();
			ArrayInt availavleActions; // contains all reward nodes "To"
			for (size_t i = 0; i < R; i++)
			{
				if (currentStateRow[i] >= 0)
				{
					availavleActions.push_back(i);
				}
			}

			return availavleActions;
		}

		static int sampleNextAction(const ArrayInt& availableActions)
		{
			RandomSelector<> selector{};
			return selector(availableActions);
		}

		void update(int currentState, int action)
		{
			const ArrayDouble& currentQLearnRow = matrixQLearn[action];

			int maxQLearnIndex;
			ArrayInt maxQLearnIndexes;
			const double maxQLearnVal = *(std::max_element(currentQLearnRow.begin(), currentQLearnRow.end()));
			for (size_t i = 0; i < currentQLearnRow.size(); ++i)
			{
				if (currentQLearnRow[i] == maxQLearnVal)
				{
					maxQLearnIndexes.push_back(i);
				}
			}

			if (maxQLearnIndexes.size() > 1)
			{
				RandomSelector<> selector{};
				maxQLearnIndex = selector(maxQLearnIndexes);
			}
			else
			{
				maxQLearnIndex = maxQLearnIndexes[0];
			}

			double maxQLearnValue = matrixQLearn[action][maxQLearnIndex];

			matrixQLearn[currentState][action] = matrixReward[currentState][action] + GAMMA * maxQLearnValue;
		}

		void addEdges(const ArrayEdge& edges)
		{
			for (size_t i = 0; i < edges.size(); ++i)
			{
				matrixGraph[edges[i].first][edges[i].second] = 1;
				matrixGraph[edges[i].second][edges[i].first] = 1;
			}

			const size_t N = matrixReward.size();
			for (size_t i = 0; i < N; ++i)
			{
				for (size_t j = 0; j < N; ++j)
				{
					if (matrixGraph[i][j] != 0)
					{
						if (i == GN)
						{
							matrixReward[i][j] = 100.0f;
						}
						else
						{
							matrixReward[i][j] = 0.0f;
						}

						if (j == GN)
						{
							matrixReward[i][j] = 100.0f;
						}
						else
						{
							matrixReward[i][j] = 0.0f;
						}
					}
				}
			}

			matrixReward[GN][GN] = 100.0f;
		}

		void printGraphNodes() const noexcept
		{
			printGenericGraphDebug<MatrixInt>(matrixGraph);
		}

		void printGraphMatrix() const noexcept
		{
			printGenericMatrixDebug<MatrixInt>(matrixGraph, " Graph ");
		}

		void printRewardMatrix() const noexcept
		{
			printGenericMatrixDebug<MatrixDouble>(matrixReward, " Reward ");
		}

		void printQLearnMatrix() const noexcept
		{
			printGenericMatrixDebug<MatrixDouble>(matrixQLearn, " QLearn ", 10);
		}

		template<typename A>
		void printArrayDebug(const A& tArray, void* excludeVal = nullptr) const noexcept
		{
			printGenericArrayDebug(tArray, excludeVal);
		}

	private:
		const uint32_t GN;
		const double GAMMA;
		MatrixInt matrixGraph;
		MatrixDouble matrixReward;
		MatrixDouble matrixQLearn;
	};
}

#endif // __RLSHORTESTPATH__