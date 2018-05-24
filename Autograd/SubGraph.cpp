
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#include "SubGraph.h"

namespace Autograd
{
	SubGraph::SubGraph(Graph *pLeft, Graph *pRight) :
		pLeft{pLeft},
		pRight{pRight}
	{
		this->beNext(pLeft);
		this->beNext(pRight);

		this->forward();
	}

	const Shape &SubGraph::shape() const
	{
		return this->pLeft->shape();
	}

	Tensor SubGraph::forward()
	{
		return this->pLeft->forward() - this->pRight->forward();
	}

	Tensor SubGraph::backwardPath(Graph *pPrev)
	{
		return this->pLeft == pPrev ? this->backward() : -this->backward();
	}
}