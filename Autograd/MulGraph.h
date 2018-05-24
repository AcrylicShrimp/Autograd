
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_MULGRAPH_H

#define _CLASS_AUTOGRAD_MULGRAPH_H

#include "Graph.h"
#include "Shape.h"
#include "Tensor.h"

#include <exception>

namespace Autograd
{
	class MulGraph final : public Graph
	{
	protected:
		Graph *pLeft;
		Graph *pRight;
		
	public:
		MulGraph(Graph *pLeft, Graph *pRight);
		MulGraph(const MulGraph &sSrc) = default;
		~MulGraph() = default;
		
	public:
		MulGraph &operator=(const MulGraph &sSrc) = default;
		
	public:
		inline Graph *left() const;
		inline Graph *right() const;
		virtual const Shape &shape() const override;
		virtual Tensor forward() override;

	protected:
		virtual Tensor backwardPath(Graph *pPrev) override;
	};

	inline Graph *MulGraph::left() const
	{
		return this->pLeft;
	}

	inline Graph *MulGraph::right() const
	{
		return this->pRight;
	}
}

#endif