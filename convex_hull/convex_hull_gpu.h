#pragma once

#include <vector>

#include "point.h"

using namespace std;

__global__ void distanceKernel(const Point *pts, int n, Point p1, Point p2,
                               int side, float *outDist) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;

  Point p = pts[gid];
  float dist = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);

  // keep only distances on the requested side; others set to 0
  outDist[gid] = (side * dist > 0) ? fabsf(dist) : 0.0f;
}

int farthestPointGPU(const vector<Point> &pts, Point p1, Point p2, int side) {
  int N = static_cast<int>(pts.size());
  if (N == 0) return -1;

  // 1. copy point array to device
  Point *d_pts;
  cudaMalloc(&d_pts, N * sizeof(Point));
  cudaMemcpy(d_pts, pts.data(), N * sizeof(Point), cudaMemcpyHostToDevice);

  // 2. output distances
  float *d_dist;
  cudaMalloc(&d_dist, N * sizeof(float));

  // 3. launch kernel
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  distanceKernel<<<blocks, threads>>>(d_pts, N, p1, p2, side, d_dist);
  cudaDeviceSynchronize();

  // 4. copy distances back
  vector<float> hDist(N);
  cudaMemcpy(hDist.data(), d_dist, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_pts);
  cudaFree(d_dist);

  // 5. CPU scan for max
  int bestIdx = -1;
  float bestD = 0.0f;
  for (int i = 0; i < N; ++i) {
    if (hDist[i] > bestD) {
      bestD = hDist[i];
      bestIdx = i;
    }
  }
  return bestIdx;  // -1 if no point on that side
}

class ConvexHullGPU {
 public:
  vector<Point> quickHull(const vector<Point> &points) {
    if (points.size() < 3) return points;

    Point minPoint = points[0];
    Point maxPoint = points[0];

    for (const auto &point : points) {
      if (point.x < minPoint.x) minPoint = point;
      if (point.x > maxPoint.x) maxPoint = point;
    }

    vector<Point> hull;
    hull.push_back(minPoint);
    findHull(points, minPoint, maxPoint, -1, hull);
    hull.push_back(maxPoint);
    findHull(points, maxPoint, minPoint, -1, hull);

    return hull;
  }

 private:
  void findHull(const vector<Point> &points, Point p1, Point p2, int side,
                vector<Point> &hull) {
    int index = farthestPointGPU(pts, p1, p2, side);  // <<< GPU call
    if (index == -1) return;

    findHull(points, points[index], p1, -side, hull);
    hull.push_back(points[index]);
    findHull(points, points[index], p2, side, hull);
  }

  float distanceToLine(Point p1, Point p2, Point p) {
    return (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
  }
};
