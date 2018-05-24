
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#include "MulGraph.h"

namespace Autograd
{
	MulGraph::MulGraph(Graph *pLeft, Graph *pRight) :
		pLeft{pLeft},
		pRight{pRight}
	{
		this->beNext(pLeft);
		this->beNext(pRight);

		this->forward();
	}

	const Shape &MulGraph::shape() const
	{
		return this->pLeft->shape();
	}

	Tensor MulGraph::forward()
	{
		return this->pLeft->forward() * this->pRight->forward();
	}

	Tensor MulGraph::backwardPath(Graph *pPrev)
	{
		return this->pLeft == pPrev ? this->pRight->forward() * this->backward() : this->pLeft->forward() * this->backward();
	}
}