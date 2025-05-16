#include <chrono>
#include <random>

#include "convex_hull.h"
#include "convex_hull_cpu.h"

using namespace std;

vector<Point> generateRandomPoints(int count, int maxCoord = 1000) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0, maxCoord);
  vector<Point> points;
  for (int i = 0; i < count; ++i) points.emplace_back(dis(gen), dis(gen));
  return points;
}

int main() {
  auto points = generateRandomPoints(1 << 24, 1 << 10);

  ConvexHullCPU chCPU;
  ConvexHull ch;

  auto start = chrono::high_resolution_clock::now();
  auto hull = ch.quickHull(points);
  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double> duration = end - start;
  cout << "QuickHull took " << duration.count() << " seconds\n";

  start = chrono::high_resolution_clock::now();
  hull = chCPU.quickHull(points);
  end = chrono::high_resolution_clock::now();

  duration = end - start;
  cout << "CPU Parallel QuickHull took " << duration.count() << " seconds\n";

  //   for (const auto& p : hull) p.print();
}
