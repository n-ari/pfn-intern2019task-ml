#pragma once

#include <cassert>
#include <cstdlib>

// structures for machine learning
namespace PFN_intern_2019{
	// original vector class for machine learning
	template<typename R>
	class vector{
	private:
		int n;
		R *arr;
	public:
		vector():vector(0){};
		vector(int n):n(n){
			assert(n >= 0);
			arr = new R[n];
			for(int i=0; i<n; i++){
				arr[i] = (R)0;
			}
		}
		vector(int n, R v):vector(n){
			for(int i=0; i<n; i++){
				arr[i] = v;
			}
		}
		vector(const vector<R> &that):vector(that.size()){
			for(int i=0; i<n; i++){
				arr[i] = that(i);
			}
		}
		vector<R> &operator=(const vector<R> &that){
			n = that.size();
			delete [] arr;
			arr = new R[n];
			for(int i=0; i<n; i++){
				arr[i] = that(i);
			}
			return *this;
		}
		~vector(){
			delete [] arr;
		}
		inline int size() const {return n;}
		R &operator()(int idx) const {
			assert(0<=idx && idx<n);
			return arr[idx];
		}

		// operator
		vector<R> operator-() const {
			vector<R> ret(*this);
			for(int i=0; i<n; i++){
				ret(i) = -ret(i);
			}
			return ret;
		}
		vector<R> &operator+=(const vector<R> &that){
			if(n==0){
				(*this) = that;
				return *this;
			}else if(that.n==0){
				return *this;
			}
			assert(n == that.size());
			for(int i=0; i<n; i++){
				arr[i] += that(i);
			}
			return *this;
		}
		vector<R> &operator-=(const vector<R> &that){
			if(n==0){
				(*this) = -that;
				return *this;
			}else if(that.n==0){
				return *this;
			}
			assert(n == that.size());
			for(int i=0; i<n; i++){
				arr[i] -= that(i);
			}
			return *this;
		}
		vector<R> &operator+=(const R &a){
			for(int i=0; i<n; i++){
				arr[i] += a;
			}
			return *this;
		}
		vector<R> &operator-=(const R &a){
			for(int i=0; i<n; i++){
				arr[i] -= a;
			}
			return *this;
		}
		vector<R> &operator*=(const R &a){
			for(int i=0; i<n; i++){
				arr[i] *= a;
			}
			return *this;
		}
		vector<R> &operator/=(const R &a){
			for(int i=0; i<n; i++){
				arr[i] /= a;
			}
			return *this;
		}

		// map
		template<typename Function>
		vector<R> map(const Function &f){
			vector<R> ret(n);
			for(int i=0; i<n; i++){
				ret(i) = f(arr[i]);
			}
			return ret;
		}
	};

	// original matrix class for machine learning
	template<typename R>
	class matrix{
	private:
		int n,m;	// n: rows, m: columns
		R *arr;
	public:
		matrix():n(0),m(0){}
		matrix(int n, int m):n(n),m(m){
			assert(n >= 0 && m >= 0);
			arr = new R[n*m];
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					arr[i*m + j] = (R)0;
				}
			}
		}
		matrix(int n, int m, R v):matrix(n,m){
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					arr[i*m + j] = v;
				}
			}
		}
		matrix(const matrix<R> &that):matrix(that.size1(),that.size2()){
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					arr[i*m + j] = that(i,j);
				}
			}
		}
		matrix<R> &operator=(const matrix<R> &that){
			n = that.size1(); m = that.size2();
			delete [] arr;
			arr = new R[n*m];
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					arr[i*m + j] = that(i,j);
				}
			}
			return *this;
		}
		~matrix(){
			delete [] arr;
		}
		inline int size1() const {return n;}
		inline int size2() const {return m;}
		R &operator()(int idx1, int idx2) const {
			assert(0<=idx1 && idx1<n && 0<=idx2 && idx2<m);
			return arr[idx1*m + idx2];
		}

		// operator
		matrix<R> operator-() const {
			matrix<R> ret(*this);
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					ret(i,j) = -(*this)(i,j);
				}
			}
			return ret;
		}
		matrix<R> &operator+=(const matrix &that){
			if(n==0 && m==0){
				(*this) = that;
				return *this;
			}else if(that.n==0 && that.m==0){
				return *this;
			}
			assert(n == that.size1() && m == that.size2());
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					(*this)(i,j) += that(i,j);
				}
			}
			return *this;
		}
		matrix<R> &operator-=(const matrix &that){
			if(n==0 && m==0){
				(*this) = that;
				return *this;
			}else if(that.n==0 && that.m==0){
				return *this;
			}
			assert(n == that.size1() && m == that.size2());
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					(*this)(i,j) -= that(i,j);
				}
			}
			return *this;
		}
		matrix<R> &operator*=(const matrix &that){
			assert(m == that.size1());
			int l = that.size2();
			matrix<R> ret(n,l);
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					for(int k=0; k<l; k++){
						ret(i,k) += (*this)(i,j) * that(j,k);
					}
				}
			}
			(*this) = ret;
			return *this;
		}
		matrix<R> &operator+=(const R &a){
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					(*this)(i,j) += a;
				}
			}
			return *this;
		}
		matrix<R> &operator-=(const R &a){
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					(*this)(i,j) -= a;
				}
			}
			return *this;
		}
		matrix<R> &operator*=(const R &a){
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					(*this)(i,j) *= a;
				}
			}
			return *this;
		}
		matrix<R> &operator/=(const R &a){
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					(*this)(i,j) /= a;
				}
			}
			return *this;
		}
		matrix<R> transpose(){
			matrix<R> ret(m,n);
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					ret(j,i) = (*this)(i,j);
				}
			}
			return ret;
		}

		// map
		template<typename Function>
		matrix<R> map(const Function &f){
			matrix<R> ret(n,m);
			for(int i=0; i<n; i++){
				for(int j=0; j<m; j++){
					ret(i,j) = f((*this)(i,j));
				}
			}
			return ret;
		}
	};

	// operator overloads
	/// vector `op` vector
	template<typename R> vector<R> operator+(const vector<R> &a, const vector<R> &b){
		return vector<R>(a) += b;
	}
	template<typename R> vector<R> operator-(const vector<R> &a, const vector<R> &b){
		return vector<R>(a) -= b;
	}
	template<typename R> R operator*(const vector<R> &a, const vector<R> &b){
		// inner product
		assert(a.size() == b.size());
		R ret = (R)0;
		for(int i=0; i<a.size(); i++){
			ret += a(i) * b(i);
		}
		return ret;
	}
	/// vector `op` real
	template<typename R> vector<R> operator+(const R &a, const vector<R> &x){
		return vector<R>(x) += a;
	}
	template<typename R> vector<R> operator+(const vector<R> &x, const R &a){
		return vector<R>(x) += a;
	}
	template<typename R> vector<R> operator-(const R &a, const vector<R> &x){
		return vector<R>(-x) += a;
	}
	template<typename R> vector<R> operator-(const vector<R> &x, const R &a){
		return vector<R>(x) -= a;
	}
	template<typename R> vector<R> operator*(const R &a, const vector<R> &x){
		return vector<R>(x) *= a;
	}
	template<typename R> vector<R> operator*(const vector<R> &x, const R &a){
		return vector<R>(x) *= a;
	}
	template<typename R> vector<R> operator/(const vector<R> &x, const R &a){
		return vector<R>(x) /= a;
	}
	/// matrix `op` matrix
	template<typename R> matrix<R> operator+(const matrix<R> &a, const matrix<R> &b){
		return matrix<R>(a) += b;
	}
	template<typename R> matrix<R> operator-(const matrix<R> &a, const matrix<R> &b){
		return matrix<R>(a) -= b;
	}
	template<typename R> matrix<R> operator*(const matrix<R> &a, const matrix<R> &b){
		return matrix<R>(a) *= b;
	}
	/// matrix `op` real
	template<typename R> matrix<R> operator+(const R &a, const matrix<R> &x){
		return matrix<R>(x) += a;
	}
	template<typename R> matrix<R> operator+(const matrix<R> &x, const R &a){
		return matrix<R>(x) += a;
	}
	template<typename R> matrix<R> operator-(const R &a, const matrix<R> &x){
		return matrix<R>(-x) += a;
	}
	template<typename R> matrix<R> operator-(const matrix<R> &x, const R &a){
		return matrix<R>(x) -= a;
	}
	template<typename R> matrix<R> operator*(const R &a, const matrix<R> &x){
		return matrix<R>(x) *= a;
	}
	template<typename R> matrix<R> operator*(const matrix<R> &x, const R &a){
		return matrix<R>(x) *= a;
	}
	template<typename R> matrix<R> operator/(const matrix<R> &x, const R &a){
		return matrix<R>(x) /= a;
	}
	/// matrix `op` vector
	template<typename R> vector<R> operator*(const matrix<R> &a, const vector<R> &b){
		assert(a.size2() == b.size());
		int n = a.size1(), m = a.size2();
		vector<R> ret(n);
		for(int i=0; i<n; i++){
			for(int j=0; j<m; j++){
				ret(i) += a(i,j) * b(j);
			}
		}
		return ret;
	}

	// advanced operators
	/// vector `op` vector
	template<typename R> vector<R> elMul(const vector<R> &a, const vector<R> &b){
		assert(a.size() == b.size());
		vector<R> ret(a.size());
		for(int i=0; i<a.size(); i++){
			ret(i) = a(i) * b(i);
		}
		return ret;
	}
	template<typename R> vector<R> elDiv(const vector<R> &a, const vector<R> &b){
		assert(a.size() == b.size());
		vector<R> ret(a.size());
		for(int i=0; i<a.size(); i++){
			ret(i) = a(i) / b(i);
		}
		return ret;
	}
	/// matrix `op` matrix
	template<typename R> matrix<R> elMul(const matrix<R> &a, const matrix<R> &b){
		assert(a.size1() == b.size1() && a.size2() == b.size2());
		matrix<R> ret(a.size1(), a.size2());
		for(int i=0; i<a.size1(); i++){
			for(int j=0; j<a.size2(); j++){
				ret(i,j) = a(i,j) * b(i,j);
			}
		}
		return ret;
	}
	template<typename R> matrix<R> elDiv(const matrix<R> &a, const matrix<R> &b){
		assert(a.size1() == b.size1() && a.size2() == b.size2());
		matrix<R> ret(a.size1(), a.size2());
		for(int i=0; i<a.size1(); i++){
			for(int j=0; j<a.size2(); j++){
				ret(i,j) = a(i,j) / b(i,j);
			}
		}
		return ret;
	}

	// graph
	// undirected edge
	template<typename R>
	class graph{
		int n;			// vertex counts
	public:
		matrix<R> g;	// graph adjacency matrix
		// constructer with no edge
		graph(int n):n(n),g(matrix<R>(n,n)){
			assert(n>0);
		}
		// load graph from adjacency matrix
		void loadGraph(matrix<R> mat){
			assert(mat.size1() == n && mat.size2() == n);
			g = mat;
		}
		// add edge
		void addEdge(int i, int j){
			assert(i>=0 && i<n && j>=0 && j<n);
			g(i,j) = g(j,i) = (R)1;
		}
		inline bool isConnected(int i, int j) const {return g(i,j);}
		inline int getN() const {return n;}
	};
}
