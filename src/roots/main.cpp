#include <iostream>
#include <iomanip>

#include "roots.hpp"

double poly1(double x)
{
    // initial bracket [-200, 300]
    return x * x * x - x * x + 2;
}

double poly1_deriv(double x)
{
    return 3 * x * x - 2 * x;
}

double poly2(double x)
{
    // initial bracket [-1,1]
    return 2 * x * x * x - 4 * x * x + 3 * x;
}

double poly2_deriv(double x)
{
    return 6 * x * x - 8 * x + 3;
}

int main(int argc, char **argv)
{
    double root = 0.0;
    const char *method_names[] = {"Bisection", "Regula Falsi", "Newton-Raphson", "Secant"};

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Testing all root-finding methods\n"
              << std::endl;

    // Test poly1 with all methods
    std::cout << "=== poly1(x) = x^3 - x^2 + 2, bracket [-200, 300] ===" << std::endl;

    if (bisection(poly1, -200.0, 300.0, &root))
    {
        std::cout << "Bisection: root = " << root << ", poly1(root) = " << poly1(root) << std::endl;
    }
    else
    {
        std::cout << "Bisection: failed" << std::endl;
    }

    if (regula_falsi(poly1, -200.0, 300.0, &root))
    {
        std::cout << "Regula Falsi: root = " << root << ", poly1(root) = " << poly1(root) << std::endl;
    }
    else
    {
        std::cout << "Regula Falsi: failed" << std::endl;
    }

    if (newton_raphson(poly1, poly1_deriv, -200.0, 300.0, -1.0, &root))
    {
        std::cout << "Newton-Raphson: root = " << root << ", poly1(root) = " << poly1(root) << std::endl;
    }
    else
    {
        std::cout << "Newton-Raphson: failed" << std::endl;
    }

    if (secant(poly1, -200.0, 300.0, -1.0, &root))
    {
        std::cout << "Secant: root = " << root << ", poly1(root) = " << poly1(root) << std::endl;
    }
    else
    {
        std::cout << "Secant: failed" << std::endl;
    }

    std::cout << std::endl;

    // Test poly2 with all methods
    std::cout << "=== poly2(x) = 2x^3 - 4x^2 + 3x, bracket [-1, 1] ===" << std::endl;

    if (bisection(poly2, -1.0, 1.0, &root))
    {
        std::cout << "Bisection: root = " << root << ", poly2(root) = " << poly2(root) << std::endl;
    }
    else
    {
        std::cout << "Bisection: failed" << std::endl;
    }

    if (regula_falsi(poly2, -1.0, 1.0, &root))
    {
        std::cout << "Regula Falsi: root = " << root << ", poly2(root) = " << poly2(root) << std::endl;
    }
    else
    {
        std::cout << "Regula Falsi: failed" << std::endl;
    }

    if (newton_raphson(poly2, poly2_deriv, -1.0, 1.0, -0.5, &root))
    {
        std::cout << "Newton-Raphson: root = " << root << ", poly2(root) = " << poly2(root) << std::endl;
    }
    else
    {
        std::cout << "Newton-Raphson: failed" << std::endl;
    }

    if (secant(poly2, -1.0, 1.0, -0.5, &root))
    {
        std::cout << "Secant: root = " << root << ", poly2(root) = " << poly2(root) << std::endl;
    }
    else
    {
        std::cout << "Secant: failed" << std::endl;
    }

    return 0;
}
