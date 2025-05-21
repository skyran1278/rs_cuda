#pragma once
#include <iostream>

struct Point {
  float x, y;

  Point(float x = 0, float y = 0) : x(x), y(y) {}

  void print() const { std::cout << "(" << x << ", " << y << ")" << std::endl; }
};
