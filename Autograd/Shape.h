
/*
	2018.05.05
	Created by AcrylicShrimp.
*/

#ifndef _CLASS_AUTOGRAD_SHAPE_H

#define _CLASS_AUTOGRAD_SHAPE_H

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace Autograd
{
	class Shape final
	{
	public:
		static constexpr int Unknown{-1};
		
	private:
		std::vector<int> sShape;
		
	public:
		Shape(std::initializer_list<int> sShape);
		Shape(const Shape &sSrc) = default;
		Shape(Shape &&sSrc) = default;
		~Shape() = default;
		
	public:
		Shape &operator=(const Shape &sSrc) = default;
		Shape &operator=(Shape &&sSrc) = default;
		bool operator==(std::initializer_list<int> sShape) const;
		bool operator==(const Shape &sSrc) const;
		bool operator!=(std::initializer_list<int> sShape) const;
		bool operator!=(const Shape &sSrc) const;
		
	public:
		inline std::size_t dimension() const;
		inline int size(std::size_t nDimension) const;
		inline int element() const;
		std::string toString() const;
	};

	inline std::size_t Shape::dimension() const
	{
		return this->sShape.size();
	}

	inline int Shape::size(std::size_t nDimension) const
	{
		return nDimension < this->dimension() ? this->sShape[nDimension] : Shape::Unknown;
	}

	int Shape::element() const
	{
		auto nResult{1};

		for (auto nSize : this->sShape)
		{
			if (nSize == Shape::Unknown)
				throw std::exception{"shape must be explicit"};

			nResult *= nSize;
		}

		return nResult;
	}
}

#endif