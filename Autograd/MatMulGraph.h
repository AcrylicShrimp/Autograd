
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_MATMULGRAPH_H

#define _CLASS_AUTOGRAD_MATMULGRAPH_H

#include "Graph.h"
#include "Shape.h"
#include "Tensor.h"

#include <exception>

namespace Autograd
{
	class MatMulGraph final : public Graph
	{
	private:
		Shape sShape;
		Graph *pLeft;
		Graph *pRight;
		
	public:
		MatMulGraph(Graph *pLeft, Graph *pRight);
		MatMulGraph(const MatMulGraph &sSrc) = default;
		~MatMulGraph() = default;
		
	public:
		MatMulGraph &operator=(const MatMulGraph &sSrc) = default;
		
	public:
		inline Graph *left() const;
		inline Graph *right() const;
		virtual const Shape &shape() const override;
		virtual Tensor forward() override;

	protected:
		virtual Tensor backwardPath(Graph *pPrev) override;
	};

	inline Graph *MatMulGraph::left() const
	{
		return this->pLeft;
	}

	inline Graph *MatMulGraph::right() const
	{
		return this->pRight;
	}
}

#endif