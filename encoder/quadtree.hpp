#ifndef QUADTREE_HPP
#define QUADTREE_HPP

#include <vector>
#include <memory>
#include <geos/geom/Envelope.h>
#include <geos/geom/Point.h>
#include <geos/geom/Polygon.h>
#include <geos/geom/Coordinate.h>
#include <geos/geom/Geometry.h>

class QuadNode {
public:
    QuadNode(const geos::geom::Envelope& bounds, size_t capacity, int level = 0)
        : bounds(bounds), capacity(capacity), level(level) {}

    // Insert a point into this node or its children.
    bool insert(const geos::geom::Coordinate* coord) {
        if (!bounds.contains(*coord)) {
            return false; // Point is not within this node's bounds.
        }

        if (points.size() < capacity && children.empty()) {
            points.push_back(coord);
            return true;
        }

        if (children.empty()) {
            subdivide();
        }

        for (auto& child : children) {
            if (child->insert(coord)) {
                return true;
            }
        }

        // This case should ideally not be reached if subdivision is correct.
        // It means the point fit in the parent but not in any child.
        return false;
    }

    const geos::geom::Envelope bounds;
    std::vector<std::unique_ptr<QuadNode>> children;
    int level;

private:
    void subdivide() {
        double minX = bounds.getMinX();
        double minY = bounds.getMinY();
        double maxX = bounds.getMaxX();
        double maxY = bounds.getMaxY();
        double midX = (minX + maxX) / 2.0;
        double midY = (minY + maxY) / 2.0;

        // Top-left
        children.push_back(std::make_unique<QuadNode>(geos::geom::Envelope(minX, midX, midY, maxY), capacity, level + 1));
        // Top-right
        children.push_back(std::make_unique<QuadNode>(geos::geom::Envelope(midX, maxX, midY, maxY), capacity, level + 1));
        // Bottom-left
        children.push_back(std::make_unique<QuadNode>(geos::geom::Envelope(minX, midX, minY, midY), capacity, level + 1));
        // Bottom-right
        children.push_back(std::make_unique<QuadNode>(geos::geom::Envelope(midX, maxX, minY, midY), capacity, level + 1));

        // Re-insert points from this node into the new children
        for (const auto* p : points) {
            for (auto& child : children) {
                if (child->insert(p)) {
                    break;
                }
            }
        }
        points.clear(); // All points have been moved to children.
    }

    size_t capacity;
    std::vector<const geos::geom::Coordinate*> points;
};

#endif // QUADTREE_HPP
