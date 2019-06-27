#pragma once

#include <cmath>

namespace PFN_intern_2019{
	// relu: max(0,x)
	template<typename R>
	R relu(R x){
		return x<(R)0 ? (R)0 : x;
	}

	// sigmoid: 1/(1+exp(-x))
	template<typename R>
	R sigmoid(R x){
		return (R)1 / ((R)1 + std::exp(-x));
	}
}
