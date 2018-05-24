
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_MEANGRAPH_H

#define _CLASS_AUTOGRAD_MEANGRAPH_H

#include "Graph.h"
#include "Shape.h"
#include "Tensor.h"

#include <numeric>

namespace Autograd
{
	class MeanGraph final : public Graph
	{
	protected:
		Shape sShape;
		Graph *pLeft;
		
	public:
		MeanGraph(Graph *pLeft);
		MeanGraph(const MeanGraph &sSrc) = default;
		~MeanGraph() = default;
		
	public:
		MeanGraph &operator=(const MeanGraph &sSrc) = default;
		
	public:
		inline Graph *left() const;
		virtual const Shape &shape() const override;
		virtual Tensor forward() override;

	protected:
		virtual Tensor backwardPath(Graph *pPrev) override;
	};

	inline Graph *MeanGraph::left() const
	{
		return this->pLeft;
	}
}

#endif