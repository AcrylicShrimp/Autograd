
/*
	2018.05.23
	Created by AcrylicShrimp.
*/

#include "AddGraph.h"

namespace Autograd
{
	AddGraph::AddGraph(Graph *pLeft, Graph *pRight) :
		pLeft{pLeft},
		pRight{pRight}
	{
		this->beNext(pLeft);
		this->beNext(pRight);

		this->forward();
	}

	const Shape &AddGraph::shape() const
	{
		return this->pLeft->shape();
	}

	Tensor AddGraph::forward()
	{
		return this->pLeft->forward() + this->pRight->forward();
	}

	Tensor AddGraph::backwardPath(Graph *pPrev)
	{
		return this->backward();
	}
}