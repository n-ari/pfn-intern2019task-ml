#pragma once

#include <cassert>

#include <iostream>

#include "functions.hpp"
#include "structures.hpp"

// graph neural network
namespace PFN_intern_2019{
	// parameter struct
	template<typename R>
	class GNNParam{
	public:
		matrix<R> W;
		vector<R> A;
		R b;
		GNNParam():GNNParam(0){}
		GNNParam(int D):W(D,D),A(D),b((R)0.0){}
		GNNParam(const matrix<R> &W, const vector<R> &A, R b):W(W),A(A),b(b){
			assert(W.size1() == W.size2());
			assert(W.size1() == A.size());
		}
		GNNParam(const GNNParam<R> &param):W(param.W),A(param.A),b(param.b){}
		GNNParam<R> &operator=(const GNNParam<R> &that){
			W = that.W;
			A = that.A;
			b = that.b;
			return *this;
		}

		// operator
		GNNParam<R> operator-() const {
			return GNNParam<R>(-W, -A, -b);
		}
		GNNParam<R> &operator+=(const GNNParam<R> &that){
			if(A.size()==0){
				(*this) = that;
				return *this;
			}else if(that.A.size()==0){
				return *this;
			}
			W += that.W;
			A += that.A;
			b += that.b;
			return *this;
		}
		GNNParam<R> &operator-=(const GNNParam<R> &that){
			if(A.size()==0){
				(*this) = -that;
				return *this;
			}else if(that.A.size()==0){
				return *this;
			}
			W -= that.W;
			A -= that.A;
			b -= that.b;
			return *this;
		}
		GNNParam<R> &operator+=(const R &a){
			W += a;
			A += a;
			b += a;
			return *this;
		}
		GNNParam<R> &operator-=(const R &a){
			W -= a;
			A -= a;
			b -= a;
			return *this;
		}
		GNNParam<R> &operator*=(const R &a){
			W *= a;
			A *= a;
			b *= a;
			return *this;
		}
		GNNParam<R> &operator/=(const R &a){
			W /= a;
			A /= a;
			b /= a;
			return *this;
		}

		// map
		template<typename Function>
		GNNParam<R> map(const Function &f){
			GNNParam<R> ret(A.size());
			ret.A = A.map(f);
			ret.W = W.map(f);
			ret.b = f(b);
			return ret;
		}
	};

	// operator overloads
	/// GNNParam `op` GNNParam
	template<typename R> GNNParam<R> operator+(const GNNParam<R> &a, const GNNParam<R> &b){
		return GNNParam<R>(a) += b;
	}
	template<typename R> GNNParam<R> operator-(const GNNParam<R> &a, const GNNParam<R> &b){
		return GNNParam<R>(a) -= b;
	}
	/// GNNParam `op` real
	template<typename R> GNNParam<R> operator+(const R &a, const GNNParam<R> &x){
		return GNNParam<R>(x) += a;
	}
	template<typename R> GNNParam<R> operator+(const GNNParam<R> &x, const R &a){
		return GNNParam<R>(x) += a;
	}
	template<typename R> GNNParam<R> operator-(const R &a, const GNNParam<R> &x){
		return GNNParam<R>(-x) += a;
	}
	template<typename R> GNNParam<R> operator-(const GNNParam<R> &x, const R &a){
		return GNNParam<R>(x) -= a;
	}
	template<typename R> GNNParam<R> operator*(const R &a, const GNNParam<R> &x){
		return GNNParam<R>(x) *= a;
	}
	template<typename R> GNNParam<R> operator*(const GNNParam<R> &x, const R &a){
		return GNNParam<R>(x) *= a;
	}
	template<typename R> GNNParam<R> operator/(const GNNParam<R> &x, const R &a){
		return GNNParam<R>(x) /= a;
	}

	// advanced operators
	/// GNNParam `op` GNNParam
	template<typename R> GNNParam<R> elMul(const GNNParam<R> &a, const GNNParam<R> &b){
		GNNParam<R> ret(a.A.size());
		ret.A = elMul(a.A, b.A);
		ret.W = elMul(a.W, b.W);
		ret.b = a.b * b.b;
		return ret;
	}
	template<typename R> GNNParam<R> elDiv(const GNNParam<R> &a, const GNNParam<R> &b){
		GNNParam<R> ret(a.A.size());
		ret.A = elDiv(a.A, b.A);
		ret.W = elDiv(a.W, b.W);
		ret.b = a.b / b.b;
		return ret;
	}

	// calculate feature vector
	template<typename R>
	vector<R> calcFeatureVector(const graph<R> &G, int T, const matrix<R> &W){
		const int D = W.size1();
		assert(W.size2() == D);
		assert(T >= 0);
		int n = G.getN();

		// initial vectors
		matrix<R> x(D,n);
		for(int i=0; i<n; i++){
			x(0,i) = (R)1;
		}

		// aggregate step
		for(int steps=1; steps<=T; steps++){
			// aggregate 1
			matrix<R> a = x * G.g;
			// aggregate 2
			x = (W*a).map(relu<R>);
		}

		// readout
		vector<R> ret(D);
		for(int i=0; i<n; i++){
			for(int j=0; j<D; j++){
				ret(j) += x(j,i);
			}
		}

		return ret;
	}

	// calculate label
	template<typename R>
	bool calcLabel(const graph<R> &G, int T, const GNNParam<R> &theta){
		vector<R> h = calcFeatureVector(G,T,theta.W);
		R v = theta.A * h + theta.b;
		return sigmoid(v) > 0.5;
	}

	// calculate loss function from data(G,y) and param
	template<typename R>
	R calcLoss(const graph<R> &G, bool y, int T, const GNNParam<R> &theta){
		vector<R> h = calcFeatureVector(G,T,theta.W);
		R v = theta.A * h + theta.b;
		if(y){
			if(-v > 100.0)return -v;
			else return std::log(1.0 + std::exp(-v));
		}else{
			if(v > 100.0)return v;
			else return std::log(1.0 + std::exp(v));
		}
	}

	// calculate gradient by numerical differentiation
	template<typename R>
	GNNParam<R> calcGradient(const graph<R> &G, bool y, int T, const GNNParam<R> &theta, R epsilon = 0.001){
		R loss = calcLoss(G,y,T,theta);
		const int D = theta.A.size();
		GNNParam<R> gradient(D);
		for(int i=0; i<D; i++){
			for(int j=0; j<D; j++){
				GNNParam<R> theta2(theta);
				theta2.W(i,j) += epsilon;
				R loss2 = calcLoss(G,y,T,theta2);
				gradient.W(i,j) = (loss2 - loss) / epsilon;
			}
		}
		for(int i=0; i<D; i++){
			GNNParam<R> theta2(theta);
			theta2.A(i) += epsilon;
			R loss2 = calcLoss(G,y,T,theta2);
			gradient.A(i) = (loss2 - loss) / epsilon;
		}
		{
			GNNParam<R> theta2(theta);
			theta2.b += epsilon;
			R loss2 = calcLoss(G,y,T,theta2);
			gradient.b = (loss2 - loss) / epsilon;
		}
		return gradient;
	}
}
