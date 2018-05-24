
/*
	2018.05.24
	Created by AcrylicShrimp.
*/

#include "MatMulGraph.h"

namespace Autograd
{
	MatMulGraph::MatMulGraph(Graph *pLeft, Graph *pRight) :
		sShape{},
		pLeft{pLeft},
		pRight{pRight}
	{
		this->beNext(pLeft);
		this->beNext(pRight);

		if (this->pLeft->shape().dimension() != 2 || this->pRight->shape().dimension() != 2)
			throw std::exception("need matrix");

		if (this->pLeft->shape().size(1) != this->pRight->shape().size(0))
			throw std::exception("shape mismatch");

		this->sShape = {this->pLeft->shape().size(0), this->pRight->shape().size(1)};
	}

	const Shape &MatMulGraph::shape() const
	{
		return this->pLeft->shape();
	}

	Tensor MatMulGraph::forward()
	{
		auto sLeft{this->pLeft->forward()};
		auto sRight{this->pRight->forward()};
		auto sResult{Tensor::zero(this->sShape)};

		for (auto nY{0}, nColumn{this->sShape.size(0)}; nY < nColumn; ++nY)
			for (auto nX{0}, nRow{this->sShape.size(1)}; nX < nRow; ++nX)
				for (auto nOffset{0}, nMaxOffset{this->pLeft->shape().size(1)}; nOffset < nMaxOffset; ++nOffset)
					sResult.data()[nY * nRow + nX] += sLeft.data()[nY * this->pLeft->shape().size(1) + nOffset] * sRight.data()[nOffset * this->pRight->shape().size(1) + nX];

		return sResult;
	}

	Tensor MatMulGraph::backwardPath(Graph *pPrev)
	{
		auto sResult{Tensor::zero(pPrev->shape())};
		auto sBackward{this->backward()};

		if (this->pLeft == pPrev)
		{
			auto sRight{this->pRight->forward()};

			for (auto nY{0}, nColumn{this->pLeft->shape().size(0)}; nY < nColumn; ++nY)
				for (auto nX{0}, nRow{this->pLeft->shape().size(1)}; nX < nRow; ++nX)
					for (auto nOffset{0}, nMaxOffset{this->pRight->shape().size(1)}; nOffset < nMaxOffset; ++nOffset)
						sResult.data()[nY * nRow + nX] += sRight.data()[nX * this->pRight->shape().size(1) + nOffset] * sBackward.data()[nY * this->sShape.size(1) + nOffset];
		}
		else
		{
			auto sLeft{this->pLeft->forward()};

			for (auto nY{0}, nColumn{this->pRight->shape().size(0)}; nY < nColumn; ++nY)
				for (auto nX{0}, nRow{this->pRight->shape().size(1)}; nX < nRow; ++nX)
					for (auto nOffset{0}, nMaxOffset{this->pLeft->shape().size(0)}; nOffset < nMaxOffset; ++nOffset)
						sResult.data()[nY * nRow + nX] += sLeft.data()[nOffset * this->pLeft->shape().size(1) + nY] * sBackward.data()[nOffset * this->sShape.size(1) + nX];
		}

		return sResult;
	}
}