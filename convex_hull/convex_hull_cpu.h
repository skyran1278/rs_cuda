#pragma once
#include <cmath>
#include <future>
#include <vector>

#include "point.h"

class ConvexHullCPU {
 public:
  vector<Point> quickHull(const vector<Point>& points) const {
    if (points.size() < 3) return points;

    Point minPoint = points[0];
    Point maxPoint = points[0];

    for (const auto& point : points) {
      if (point.x < minPoint.x) minPoint = point;
      if (point.x > maxPoint.x) maxPoint = point;
    }

    vector<Point> hull;
    hull.push_back(minPoint);

    // Use async to parallelize
    auto futureLeft = async(launch::async, &ConvexHullCPU::findHull, this,
                            cref(points), minPoint, maxPoint, -1);
    auto futureRight = async(launch::async, &ConvexHullCPU::findHull, this,
                             cref(points), maxPoint, minPoint, -1);

    auto left = futureLeft.get();
    auto right = futureRight.get();

    hull.insert(hull.end(), left.begin(), left.end());
    hull.push_back(maxPoint);
    hull.insert(hull.end(), right.begin(), right.end());

    return hull;
  }

 private:
  vector<Point> findHull(const vector<Point>& points, Point p1, Point p2,
                         int side) const {
    int index = -1;
    float maxDistance = 0;

    for (int i = 0; i < points.size(); i++) {
      float dist = distanceToLine(p1, p2, points[i]);
      if (side * dist > 0 && fabs(dist) > maxDistance) {
        index = i;
        maxDistance = fabs(dist);
      }
    }

    if (index == -1) return {};

    auto left = async(launch::async, &ConvexHullCPU::findHull, this,
                      cref(points), points[index], p1, -side);
    auto right = async(launch::async, &ConvexHullCPU::findHull, this,
                       cref(points), points[index], p2, side);

    auto leftResult = left.get();
    auto rightResult = right.get();

    leftResult.push_back(points[index]);
    leftResult.insert(leftResult.end(), rightResult.begin(), rightResult.end());

    return leftResult;
  }

  float distanceToLine(Point p1, Point p2, Point p) const {
    return (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
  }
};
