
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#include "SquareGraph.h"

namespace Autograd
{
	SquareGraph::SquareGraph(Graph *pLeft) :
		pLeft{pLeft}
	{
		this->beNext(pLeft);

		this->forward();
	}

	const Shape &SquareGraph::shape() const
	{
		return this->pLeft->shape();
	}

	Tensor SquareGraph::forward()
	{
		return this->pLeft->forward() * this->pLeft->forward();
	}

	Tensor SquareGraph::backwardPath(Graph *pPrev)
	{
		return this->pLeft->forward() * this->backward();
	}
}