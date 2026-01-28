#include <gtest/gtest.h>
#include <cmath>
#include "roots.hpp"

TEST(Bisection, FindsPositiveRoot)
{
    auto f = [](double x)
    { return x * x - 4.0; };
    double root = 0.0;
    ASSERT_TRUE(bisection(f, 0.0, 3.0, &root));
    EXPECT_NEAR(root, 2.0, 1e-5);
}

TEST(Bisection, ReturnsFalseWithoutSignChange)
{
    auto f = [](double x)
    { return x * x + 1.0; };
    double root = 0.0;
    EXPECT_FALSE(bisection(f, 0.0, 1.0, &root));
}

TEST(RegulaFalsi, ConvergesOnRoot)
{
    auto f = [](double x)
    { return x * x - 9.0; };
    double root = 0.0;
    ASSERT_TRUE(regula_falsi(f, 0.0, 5.0, &root));
    EXPECT_NEAR(root, 3.0, 1e-4);
}

TEST(NewtonRaphson, ConvergesQuadratic)
{
    auto f = [](double x)
    { return x * x - 4.0; };
    auto df = [](double x)
    { return 2.0 * x; };
    double root = 0.0;
    ASSERT_TRUE(newton_raphson(f, df, 0.0, 3.0, 1.0, &root));
    EXPECT_NEAR(root, 2.0, 1e-5);
}

TEST(Secant, ConvergesQuadratic)
{
    auto f = [](double x)
    { return x * x - 4.0; };
    double root = 0.0;
    ASSERT_TRUE(secant(f, 0.0, 3.0, 1.0, &root));
    EXPECT_NEAR(root, 2.0, 1e-4);
}