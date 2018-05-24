
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_SUBGRAPH_H

#define _CLASS_AUTOGRAD_SUBGRAPH_H

#include "Graph.h"
#include "Shape.h"
#include "Tensor.h"

#include <exception>

namespace Autograd
{
	class SubGraph final : public Graph
	{
	protected:
		Graph *pLeft;
		Graph *pRight;
		
	public:
		SubGraph(Graph *pLeft, Graph *pRight);
		SubGraph(const SubGraph &sSrc) = default;
		~SubGraph() = default;
		
	public:
		SubGraph &operator=(const SubGraph &sSrc) = default;
		
	public:
		inline Graph *left() const;
		inline Graph *right() const;
		virtual const Shape &shape() const override;
		virtual Tensor forward() override;

	protected:
		virtual Tensor backwardPath(Graph *pPrev) override;
	};

	inline Graph *SubGraph::left() const
	{
		return this->pLeft;
	}

	inline Graph *SubGraph::right() const
	{
		return this->pRight;
	}
}

#endif