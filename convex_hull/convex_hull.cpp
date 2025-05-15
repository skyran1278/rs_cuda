#include <iostream>
#include <vector>

using namespace std;

struct Point {
  float x, y;
  Point(float x, float y) : x(x), y(y) {}
  Point() : x(0), y(0) {}

  void print() const { cout << "(" << x << ", " << y << ")" << endl; }
};

class ConvexHull {
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
    int index = -1;
    float maxDistance = 0;

    for (int i = 0; i < points.size(); i++) {
      float dist = distanceToLine(p1, p2, points[i]);
      if (side * dist > 0 && fabs(dist) > maxDistance) {
        index = i;
        maxDistance = fabs(dist);
      }
    }

    if (index == -1) return;

    findHull(points, points[index], p1, -side, hull);
    hull.push_back(points[index]);
    findHull(points, points[index], p2, side, hull);
  }

  float distanceToLine(Point p1, Point p2, Point p) {
    return (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
  }
};

int main() {
  ConvexHull ch;
  vector<Point> point1s = {Point(0, 0),  Point(1, 1),     Point(2, 0),
                           Point(1, -1), Point(0.5, 0.5), Point(1.5, -0.5)};
  vector<Point> point2s = {Point(0, 0),  Point(2, 2), Point(4, 0),
                           Point(2, 1),  Point(3, 2),  // interior point
                           Point(1, -1), Point(3, -1)};
  vector<Point> hull = ch.quickHull(point2s);

  for (const auto &point : hull) {
    point.print();
  }

  return 0;
}
