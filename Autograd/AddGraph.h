
/*
	2018.05.23
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_ADDGRAPH_H

#define _CLASS_AUTOGRAD_ADDGRAPH_H

#include "Graph.h"
#include "Shape.h"
#include "Tensor.h"

#include <exception>

namespace Autograd
{
	class AddGraph final : public Graph
	{
	protected:
		Graph *pLeft;
		Graph *pRight;
		
	public:
		AddGraph(Graph *pLeft, Graph *pRight);
		AddGraph(const AddGraph &sSrc) = default;
		~AddGraph() = default;
		
	public:
		AddGraph &operator=(const AddGraph &sSrc) = default;
		
	public:
		inline Graph *left() const;
		inline Graph *right() const;
		virtual const Shape &shape() const override;
		virtual Tensor forward() override;

	protected:
		virtual Tensor backwardPath(Graph *pPrev) override;
	};

	inline Graph *AddGraph::left() const
	{
		return this->pLeft;
	}

	inline Graph *AddGraph::right() const
	{
		return this->pRight;
	}
}

#endif