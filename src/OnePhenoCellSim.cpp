#include <cmath>

#include "Applications/Definitions/functors.hpp"
#include "Applications/NumericalIntegration/GaussLegendre.hpp"
#include "Applications/ODEIntegration/RungeKutta45.hpp"
#include "Libraries/Vector/vectorBase.hpp"

struct MorseKernel : public RtoR<real> {
	real R, A, r, a;
	MorseKernel(real R, real A, real r, real a) : R(R), A(A), r(r), a(a) {}
	real eval(real x) const override {
		return R * exp(abs(x / r)) - A * exp(abs(x / a));
	}
};

struct EuclideanDistance : public XYtoR<real> {
	EuclideanDistance() {};
	double eval(double x, double y) const override {
		x *= x;
		y *= y;
		return sqrt(x + y);
	}
};

struct ChemicalCue : public XYtoR<real> {
	double eval(double x, double y) const override { return 5 + 0.01 * x; }
};

struct Kernel : public RtoR<real> {
	Kernel() {};
	double eval(double alpha) const override {
		return 0.75 * pow(alpha, 3) - 0.25 * pow(alpha, 4);
	}
};

void normalizeXY(real *x, uint halfN) {
	for (uint i = 0; i < halfN; i++) {
		real xi = x[2 * i];
		real yi = x[2 * i + 1];
		real norm = std::hypot(xi, yi);
		x[2 * i] /= norm;
		x[2 * i + 1] /= norm;
	}
}

struct SWModel : public ODEIntegration::ODEDynamicsMatrix<real, 0, 2> {
   public:
	using MatrixX2 = LinAlgebra::Matrix<real, 0, 2>;

	// prelims
	MorseKernel morse;
	EuclideanDistance euclDist;
	ChemicalCue cue;
	Kernel kernel;

	const uint numCells;
	const real cellRadius = 10.0;
	const real compDomainExtent = 500.0;
	real morseCueWeight = 0.5;
	MatrixX2 state;

	SWModel()
		: numCells(100), morse(MorseKernel(1, 1, 10, 20)) {
			  // TODO: Initialize initial state in some way

		  };
	~SWModel() {};

	void PreIntegration(MatrixX2 &x, real t) override {}

	void PostIntegration(MatrixX2 &x, real t) override {
		normalizeXY(x(x.idx(0, 1)), numCells);
	}

	double stateNorm(const MatrixX2 &x) override {
		real norminf = 0.0;
		for (uint i = 0; i < x.size(); i++) {
			norminf = std::max(norminf, std::abs(x[i]));
		}
		return norminf / compDomainExtent;
	}

	LinAlgebra::Matrix<real, 0, 0> distanceMatrix;
	LinAlgebra::Vector<real, 0> morseForces;
	LinAlgebra::Vector<real, 0> envForces;
	void Gradient(const MatrixX2 &x, MatrixX2 &gradout, real t) override {
		distanceMatrix.alloc(numCells, numCells);

		gradout.setZero();
		distanceMatrix.setZero();
		envForces.setZero();
		morseForces.setZero();

		// pair-wise computations for distance, morse interactions
		const real *positions = &x(0, 0);
		const real *polarities = &x(0, 1);
		real *dpos = &gradout(0, 0);
		real *dpolar = &gradout(0, 1);
		for (uint jCell = 0; jCell < numCells; jCell++) {
			real xj = positions[jCell * 2];
			real yj = positions[jCell * 2 + 1];
			real xMorse = 0.0;
			real yMorse = 0.0;

			real *distJ = &distanceMatrix(0, jCell);
			for (uint iCell = 0; iCell < jCell; iCell++) {
				real xi = positions[iCell * 2];
				real yi = positions[iCell * 2 + 1];

				real xdiff = xj - xi;
				real ydiff = yj - yi;
				real dist = euclDist.eval(xdiff, ydiff);
				real fMorse = morse.eval(dist);

				distJ[iCell] = dist;
				xMorse += xdiff * fMorse;
				yMorse += ydiff * fMorse;
			}

			for (uint iCell = jCell + 1; iCell < numCells; iCell++) {
				real xi = positions[iCell * 2];
				real yi = positions[iCell * 2 + 1];

				real xdiff = xj - xi;
				real ydiff = yj - yi;
				real dist = euclDist.eval(xdiff, ydiff);
				real fMorse = morse.eval(dist);

				distJ[iCell] = dist;
				xMorse += xdiff * fMorse;
				yMorse += ydiff * fMorse;
			}
			morseForces[jCell * 2] = xMorse;
			morseForces[jCell * 2 + 1] = yMorse;
		}

		// environment force calculations
		constexpr uint numGLPoints = 64;
		const real *glwt32 = Quadrature::GaussLegendre<real>::wt64;
		const real *glab32 = Quadrature::GaussLegendre<real>::wt64;

		real A = Quadrature::GaussLegendre<real>::eval<Kernel, 64>(kernel);
		for (uint iCell = 0; iCell < numCells; iCell++) {
			real xi = positions[iCell * 2];
			real yi = positions[iCell * 2 + 1];

			real xEnv = 0.0;
			real yEnv = 0.0;
			real cueXiYi = cue.eval(xi, yi);
			for (uint iTheta = 0; iTheta < numGLPoints / 2; iTheta++) {
				real theta = glab32[iTheta] * 2.0 * M_PI;
				real dx = cellRadius * cos(theta);
				real dy = cellRadius * sin(theta);
				real cueXY = cue.eval(xi + dx, yi + dy);

				real kernelFactor = (cueXY - cueXiYi) * kernel.eval(theta) / A;
				xEnv += glwt32[iTheta] * dx * kernelFactor;
				yEnv += glwt32[iTheta] * dy * kernelFactor;
			}
			envForces[iCell * 2] = xEnv;
			envForces[iCell * 2 + 1] = yEnv;
		}
		normalizeXY(envForces(), numCells);

		for (uint i = 0; i < numCells * 2; i++) {
			dpolar[i] += morseCueWeight * morseForces[i] +
						 (1 - morseCueWeight) * envForces[i];
		}
	}
};

int main() {
	SWModel model;
	ODEIntegration::RungeKutta45Matrix integrator(model, 0.001, 1e-4, 2, 1e-5,
												  1e-5);
	return 0;
}
