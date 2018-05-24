
/*
	2018.05.05
	Created by AcrylicShrimp.
*/

#include "AddGraph.h"
#include "MatMulGraph.h"
#include "MeanGraph.h"
#include "MulGraph.h"
#include "Shape.h"
#include "SquareGraph.h"
#include "SubGraph.h"
#include "Tensor.h"
#include "ValueGraph.h"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace Autograd;

int main()
{
	ValueGraph x{Shape{2, 1}};
	ValueGraph y{Shape{1}};

	ValueGraph w{Shape{1, 2}};
	ValueGraph b{Shape{1}};

	auto wx{MatMulGraph(&w, &x)};
	auto wx_b{AddGraph(&wx, &b)};

	auto y_wx_b{SubGraph(&y, &wx_b)};
	auto y_wx_b_sqr{SquareGraph(&y_wx_b)};
	auto loss{MeanGraph(&y_wx_b_sqr)};

	std::mt19937_64 sEngine{std::random_device{}()};
	std::uniform_real<float> sRandom{-1.f, 1.f};

	w.value() = {sRandom(sEngine), sRandom(sEngine)};
	b.value() = {.0f};

	for (;;)
	{
		system("cls");

		x.value() = {.0f, .0f};
		y.value() = {.0f};
		printf("wx_b :\n%s\n\n", wx_b.forward().toString().c_str());
		printf("loss :\n%s\n\n", loss.forward().toString().c_str());
		printf("weight_gradient :\n%s\n\n", w.backward().toString().c_str());
		printf("bias_gradient :\n%s\n\n", b.backward().toString().c_str());

		puts("============================================");

		x.value() = {1.f, .0f};
		y.value() = {1.f};
		printf("wx_b :\n%s\n\n", wx_b.forward().toString().c_str());
		printf("loss :\n%s\n\n", loss.forward().toString().c_str());
		printf("weight_gradient :\n%s\n\n", w.backward().toString().c_str());
		printf("bias_gradient :\n%s\n\n", b.backward().toString().c_str());

		puts("============================================");

		x.value() = {.0f, 1.f};
		y.value() = {1.f};
		printf("wx_b :\n%s\n\n", wx_b.forward().toString().c_str());
		printf("loss :\n%s\n\n", loss.forward().toString().c_str());
		printf("weight_gradient :\n%s\n\n", w.backward().toString().c_str());
		printf("bias_gradient :\n%s\n\n", b.backward().toString().c_str());

		puts("============================================");

		x.value() = {1.f, 1.f};
		y.value() = {1.f};
		printf("wx_b :\n%s\n\n", wx_b.forward().toString().c_str());
		printf("loss :\n%s\n\n", loss.forward().toString().c_str());
		printf("weight_gradient :\n%s\n\n", w.backward().toString().c_str());
		printf("bias_gradient :\n%s\n\n", b.backward().toString().c_str());

		system("pause");

		{
			x.value() = {.0f, .0f};
			y.value() = {.0f};
			
			auto gradient_w{w.backward()};
			auto gradient_b{b.backward()};

			w.value() -= gradient_w * .1f;
			b.value() -= gradient_b * .1f;
		}

		{
			x.value() = {1.f, .0f};
			y.value() = {1.f};

			auto gradient_w{w.backward()};
			auto gradient_b{b.backward()};

			w.value() -= gradient_w * .1f;
			b.value() -= gradient_b * .1f;
		}

		{
			x.value() = {.0f, 1.f};
			y.value() = {1.f};

			auto gradient_w{w.backward()};
			auto gradient_b{b.backward()};

			w.value() -= gradient_w * .1f;
			b.value() -= gradient_b * .1f;
		}

		{
			x.value() = {1.f, 1.f};
			y.value() = {1.f};

			auto gradient_w{w.backward()};
			auto gradient_b{b.backward()};

			w.value() -= gradient_w * .1f;
			b.value() -= gradient_b * .1f;
		}
	}

	return 0;
}