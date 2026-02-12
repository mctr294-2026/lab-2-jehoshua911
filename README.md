# root-finding

## Introduction

You will find that much of the theory in your undergraduate degree involves the discovery and application of analytical equations that model problems of interest. However, there exist large classes of problems where analytical solutions simply do not exist or are otherwise intractable due to complexity/scope. These problems can often be easily solved via methods of numerical computation and can provide us with usable solutions with precision limited only by computational power and our algorithms. Numerical computation is a course in and of itself, but here we will touch on a basic, but common problem; root-finding.

Root finding can be described as finding all the zero-crossings of an analytical function. By applying root finding to 1st or 2nd derivatives one can also find minima, maxima and inflection points of analytical functions.  For our purposes we will define root-finding as follows: If, for a given function `f(x)` defined at all points on the interval `x ∈ [a, b]` and `f(x) = 0` has 0 or 1 solutions for `x ∈ [a, b]`, find the value of x if it exists. Most root finders algorithms work by iteratively evaluating `f(x)` at various points on the interval `[a, b]` until they can estimate `x` within some predefined precision.

## Requirements

- Provided a function pointer that evaluates `f(x)`
  - Implement several root finding algorithms that can find the root of `f(x)`
  - Utilize these root finding algorithm to find local maxima of `f(x)`
  
- The required root-finding algorithms are listed here in (approximately) increasing order of difficulty
  - Bisection Method
  - Regula Falsi
  - Newton-Raphson's Method (you will also be given a function pointer that evaluates `g(x) = f'(x)`)
  - Secant Method

### Testing Requirements

- For the sake of testing we require that all your source files are placed in `src/roots`.
- We also require that nothing in `src/roots/roots.hpp`. You may feel free to add to this file or add additional files any directories, as long as testing continues to build and run. You may make changes to `CMakeLists.txt` to accomodate any changes that surpass these bounds, but you are responsible for confirming that testing continues to work.
- Your functions are expected to return the correct root within a tolerance of 1e-6. You can also limit iterations to 1e6 to guarantee completion.
- The tests provided are not exhaustive and during marking more tests may be added, to confirm your code works according to specifications. You are encouraged to add tests of your own to vet your implementations.

### Hints

- The required algorithms are described [here](https://www.geeksforgeeks.org/maths/root-finding-algorithm/).
- We have provided a sample program that can act as a test driver in `src/roots/main.cpp` but the resulting binary will not be tested. You can modify this program or add additional tests to the `tests` folder to confirm that your implementations function correctly.

## Building

```shell
cmake -S . -B build
cmake --build build --config Debug
build\Debug\roots_demo.exe
```

## Testing

```shell
ctest --test-dir build -C Debug
Add -V for verbose testing
Add --output-on-failure for information on failed tests
```

## Assignment

After you have finished these steps switch to the `roots` branch for part 2 of this lab

``` shell
git switch roots
```
