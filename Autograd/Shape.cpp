
/*
	2018.05.05
	Created by AcrylicShrimp.
*/

#include "Shape.h"

namespace Autograd
{
	Shape::Shape(std::initializer_list<int> sShape) :
		sShape{sShape.begin(), sShape.end()}
	{
		//Empty.
	}

	bool Shape::operator==(std::initializer_list<int> sShape) const
	{
		if (this->dimension() != sShape.size())
			return false;

		const auto pShape{sShape.begin()};

		for (std::size_t nDimension{0}, nMaxDimension{this->dimension()}; nDimension < nMaxDimension; ++nDimension)
			if (this->size(nDimension) != pShape[nDimension])
				return false;

		return true;
	}

	bool Shape::operator==(const Shape &sSrc) const
	{
		if (this->dimension() != sSrc.dimension())
			return false;

		for (std::size_t nDimension{0}, nMaxDimension{this->dimension()}; nDimension < nMaxDimension; ++nDimension)
			if (this->size(nDimension) != sSrc.size(nDimension))
				return false;

		return true;
	}

	bool Shape::operator!=(std::initializer_list<int> sShape) const
	{
		if (this->dimension() != sShape.size())
			return true;

		const auto pShape{sShape.begin()};

		for (std::size_t nDimension{0}, nMaxDimension{this->dimension()}; nDimension < nMaxDimension; ++nDimension)
			if (this->size(nDimension) != pShape[nDimension])
				return true;

		return false;
	}

	bool Shape::operator!=(const Shape &sSrc) const
	{
		if (this->dimension() != sSrc.dimension())
			return true;

		for (std::size_t nDimension{0}, nMaxDimension{this->dimension()}; nDimension < nMaxDimension; ++nDimension)
			if (this->size(nDimension) != sSrc.size(nDimension))
				return true;

		return false;
	}

	std::string Shape::toString() const
	{
		std::string sResult{"["};

		for (std::size_t nDimension{0}, nMaxDimension{this->dimension()}; nDimension < nMaxDimension; ++nDimension)
		{
			if (nDimension)
				sResult += ", ";

			sResult += std::to_string(this->size(nDimension));
		}

		return sResult += "]";
	}
}