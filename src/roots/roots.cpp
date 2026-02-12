#include "roots.hpp"

bool bisection(std::function<double(double)> f,
               double a, double b,
               double *root) {
    return false;
}

bool regula_falsi(std::function<double(double)> f,
                  double a, double b,
                  double *root) {
    return false;
}

bool newton_raphson(std::function<double(double)> f,
                    std::function<double(double)> g,
                    double a, double b, double c,
                    double *root) {
    return false;
}

bool secant(std::function<double(double)> f,
            double a, double b, double c,
            double *root) {
    return false;
}

