#include "roots.hpp"
#include <cmath>        // to apply functions: abs(), isfinite()
// Convergence tolerance function
static constexpr double TOL = 1e-6;
// Provides a smaller number so we can avoid division by 0
static constexpr double EPS = 1e-14;
// Provides a max number of iteration to prevent inf loops
static constexpr std::size_t MAX_ITERS = 1'000'000;
// The below function helps to check for convergence
// where    fx : function value f(x)  and,
//          dx : change in x between iterations
// It returns true if the solution is considered to be converged
static bool done(double fx, double dx = 0.0)
{
    return std::abs(fx) <= TOL || std::abs(dx) <= TOL;
}
//Finds a root by repeatedly cutting an interval in half
bool bisection(std::function<double(double)> f,
               double a, double b,
               double* root)
{
    // This Checks the output pointer
    if (!root) return false;
    // For ensuring a <= b
    if (a > b) std::swap(a, b);
    // Evaluates the function at the endpoints
    double fa = f(a);
    double fb = f(b);
    // Checks if endpoints are already roots
    if (done(fa)) { *root = a; return true; }
    if (done(fb)) { *root = b; return true; }
    // Bisection requires a sign change
    if (fa * fb > 0) return false;
    // Iterative bisection
    for (std::size_t i = 0; i < MAX_ITERS; ++i) {
        // Midpoint
        double c = 0.5 * (a + b);
        double fc = f(c);
        // Convergence check
        if (done(fc, b - a)) {
            *root = c;
            return true;
        }
        // Keep the subinterval with sign change
        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    return false;
}
//Finds a root using a straight-line approximation instead of a midpoint
bool regula_falsi(std::function<double(double)> f,
                  double a, double b,
                  double* root)
{
    if (!root) return false;
    if (a > b) std::swap(a, b);
    double fa = f(a);
    double fb = f(b);
    // Endpoint checks
    if (done(fa)) { *root = a; return true; }
    if (done(fb)) { *root = b; return true; }
    if (fa * fb > 0) return false;
    for (std::size_t i = 0; i < MAX_ITERS; ++i) {
        double denom = fb - fa;
        if (std::abs(denom) <= EPS) return false;
        double c = (a * fb - b * fa) / denom;
        double fc = f(c);
        // Convergence check
        if (done(fc, b - a)) {
            *root = c;
            return true;
        }
        // Maintain sign-changing interval
        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    return false;
}
//Finds a root using tangent lines
bool newton_raphson(std::function<double(double)> f,
                    std::function<double(double)> g,
                    double a, double b,
                    double x,
                    double* root)
{
    if (!root) return false;
    if (a > b) std::swap(a, b);
    if (x < a || x > b) return false;
    for (std::size_t i = 0; i < MAX_ITERS; ++i) {
        double fx = f(x);
        // Check convergence using function value
        if (done(fx)) {
            *root = x;
            return true;
        }
        double gx = g(x);
        // Protect against division by zero
        if (std::abs(gx) <= EPS) return false;
        double x_next = x - fx / gx;
        // Fail if iteration leaves interval or is invalid
        if (!std::isfinite(x_next) || x_next < a || x_next > b)
            return false;
        // Check convergence using step size
        if (done(0.0, x_next - x)) {
            *root = x_next;
            return true;
        }
        x = x_next;
    }
    return false;
}
// Finds a root using secant line approximation
bool secant(std::function<double(double)> f,
            double a, double b,
            double c,
            double* root)
{
    if (!root) return false;
    if (a > b) std::swap(a, b);
    // Initial guess must lie inside interval
    if (c < a || c > b) return false;
    double x0 = a, x1 = b;
    double f0 = f(x0), f1 = f(x1);
    double fc = f(c);
    if (done(f0)) { *root = x0; return true; }
    if (done(f1)) { *root = x1; return true; }
    if (done(fc)) { *root = c;  return true; }
    // Choose better starting pair using c
    if (f0 * fc < 0) {
        x1 = c; f1 = fc;
    } else if (f1 * fc < 0) {
        x0 = c; f0 = fc;
    }
    for (std::size_t i = 0; i < MAX_ITERS; ++i) {
        double denom = f1 - f0;
        if (std::abs(denom) <= EPS) return false;
        // Secant formula
        double x2 = x1 - f1 * (x1 - x0) / denom;
        if (!std::isfinite(x2) || x2 < a || x2 > b)
            return false;
        double f2 = f(x2);
        if (done(f2, x2 - x1)) {
            *root = x2;
            return true;
        }
        // Shift points
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f2;
    }
    return false;
}