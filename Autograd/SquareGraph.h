
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_SQUAREGRAPH_H

#define _CLASS_AUTOGRAD_SQUAREGRAPH_H

#include "Graph.h"
#include "Shape.h"
#include "Tensor.h"

namespace Autograd
{
	class SquareGraph final : public Graph
	{
	protected:
		Graph *pLeft;
		
	public:
		SquareGraph(Graph *pLeft);
		SquareGraph(const SquareGraph &sSrc) = default;
		~SquareGraph() = default;
		
	public:
		SquareGraph &operator=(const SquareGraph &sSrc) = default;
		
	public:
		inline Graph *left() const;
		virtual const Shape &shape() const override;
		virtual Tensor forward() override;

	protected:
		virtual Tensor backwardPath(Graph *pPrev) override;
	};

	inline Graph *SquareGraph::left() const
	{
		return this->pLeft;
	}
}

#endif