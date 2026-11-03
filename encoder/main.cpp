#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "utils.hpp" // new CSV reader
#include "quadtree.hpp"

#include <geos/geom/Geometry.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Polygon.h>
#include <geos/io/WKTReader.h>
#include <geos/util/GEOSException.h>
#include <geos/geom/Point.h>
#include <geos/geom/util/AffineTransformation.h>
#include <geos/geom/Envelope.h>
#include <geos/index/strtree/STRtree.h>
#include <geos/geom/prep/PreparedGeometryFactory.h>
#include <geos/geom/prep/PreparedGeometry.h>

// Callback for STRtree query
class STRtreeVisitor : public geos::index::ItemVisitor {
public:
    std::vector<size_t> found_indices;
    void visitItem(void* item) override {
        found_indices.push_back(reinterpret_cast<size_t>(item));
    }
};


std::unique_ptr<geos::geom::Geometry> centerAndScalePolygon(
    const geos::geom::Geometry* poly,
    const geos::geom::GeometryFactory* factory) {

    double area = poly->getArea();
    if (area <= 1e-9) return nullptr;

    std::unique_ptr<geos::geom::Point> centroid(poly->getCentroid());
    if (!centroid) return nullptr;

    double scale_factor = 1.0 / sqrt(area);
    geos::geom::util::AffineTransformation transform;
    transform.translate(-centroid->getX(), -centroid->getY());
    transform.scale(scale_factor, scale_factor);

    return transform.transform(poly);
}


std::vector<std::unique_ptr<geos::geom::Geometry>> centerAndScalePolygons(
    const std::vector<std::unique_ptr<geos::geom::Geometry>>& polygons,
    const geos::geom::GeometryFactory* factory) {

    std::cout << "Centering and scaling polygons..." << std::endl;
    std::vector<std::unique_ptr<geos::geom::Geometry>> normalized_polygons;
    normalized_polygons.reserve(polygons.size());

    for (const auto& poly_ptr : polygons) {
        auto transformed_poly = centerAndScalePolygon(poly_ptr.get(), factory);
        if (transformed_poly) {
            normalized_polygons.push_back(std::move(transformed_poly));
        }
    }
    std::cout << "Finished centering and scaling " << normalized_polygons.size() << " polygons." << std::endl;
    return normalized_polygons;
}

void getLeafNodes(const QuadNode* node, std::vector<geos::geom::Envelope>& leaf_envelopes, std::vector<int>& leaf_levels) {
    if (node == nullptr) {
        return;
    }

    if (node->children.empty()) {
        // This is a leaf node
        leaf_envelopes.push_back(node->bounds);
        leaf_levels.push_back(node->level);
    } else {
        // Recursively traverse children
        for (const auto& child : node->children) {
            getLeafNodes(child.get(), leaf_envelopes, leaf_levels);
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_polygon_csv> <output_vector_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::cout << "Reading polygons from: " << input_file << std::endl;

    // Get the GEOS geometry factory and a WKT reader
    auto geomFactory = geos::geom::GeometryFactory::create();
    geos::io::WKTReader reader(*geomFactory);

    // --- 1. Build the grid from a sample ---
    std::vector<std::unique_ptr<geos::geom::Geometry>> polygons;
    try {
        CSVReader csv_reader(input_file);
        PolygonData data;
        int valid_polygons = 0;
        const int sample_size = 10000;

        while (csv_reader.next_row(data) && valid_polygons < sample_size) {
            try {
                std::unique_ptr<geos::geom::Geometry> geom(reader.read(data.wkt));
                if (geom && geom->getGeometryTypeId() == geos::geom::GEOS_POLYGON && geom->isValid()) {
                    polygons.push_back(std::move(geom));
                    valid_polygons++;
                }
            } catch (const geos::util::GEOSException& e) {}
        }
        std::cout << "Total valid polygons in sample: " << polygons.size() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error reading sample: " << e.what() << std::endl;
        return 1;
    }

    auto normalized_polygons = centerAndScalePolygons(polygons, geomFactory.get());
    polygons.clear(); // Free memory

    geos::geom::Envelope total_bounds;
    for (const auto& poly : normalized_polygons) {
        total_bounds.expandToInclude(poly->getEnvelopeInternal());
    }
    std::cout << "Total bounding box of normalized polygons: " << total_bounds.toString() << std::endl;

    const size_t quadtree_capacity = 100;
    auto quadtree_root = std::make_unique<QuadNode>(total_bounds, quadtree_capacity);
    for (const auto& poly_geom : normalized_polygons) {
        std::unique_ptr<geos::geom::CoordinateSequence> coords(poly_geom->getCoordinates());
        for (size_t i = 0; i < coords->size(); ++i) {
            quadtree_root->insert(&((*coords)[i]));
        }
    }
    normalized_polygons.clear(); // Free memory

    std::vector<geos::geom::Envelope> leaf_envelopes;
    std::vector<int> leaf_levels;
    getLeafNodes(quadtree_root.get(), leaf_envelopes, leaf_levels);
    size_t vector_dimension = leaf_envelopes.size();
    std::cout << "Quadtree traversal complete. Found " << vector_dimension << " leaf nodes (grid cells)." << std::endl;

    geos::index::strtree::STRtree rtree;
    for (size_t i = 0; i < leaf_envelopes.size(); ++i) {
        rtree.insert(&leaf_envelopes[i], (void*)i);
    }
    rtree.build();
    std::cout << "Finished building R-tree." << std::endl;

    std::vector<std::unique_ptr<geos::geom::Geometry>> grid_cell_polygons;
    grid_cell_polygons.reserve(leaf_envelopes.size());
    for(const auto& env : leaf_envelopes) {
        grid_cell_polygons.push_back(geomFactory->toGeometry(&env));
    }

    // --- 2. Process the full dataset in chunks ---
    std::cout << "\nProcessing full dataset to generate vectors..." << std::endl;
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }
    // Write temporary header, will be updated later
    out_file << "0 " << vector_dimension << "\n";
    long long database_vectors_written = 0;

    try {
        CSVReader csv_reader(input_file); // Re-open from the start
        PolygonData data;
        auto prepared_geom_factory = geos::geom::prep::PreparedGeometryFactory::create();

        while (csv_reader.next_row(data)) {
            std::unique_ptr<geos::geom::Geometry> original_poly(reader.read(data.wkt));
            if (!original_poly || original_poly->getGeometryTypeId() != geos::geom::GEOS_POLYGON || !original_poly->isValid()) {
                continue;
            }

            double original_area = original_poly->getArea();
            if (original_area <= 1e-9) continue;

            auto scaled_poly = centerAndScalePolygon(original_poly.get(), geomFactory.get());
            if (!scaled_poly) continue;

            auto prepared_poly = prepared_geom_factory->create(scaled_poly.get());

            STRtreeVisitor visitor;
            rtree.query(scaled_poly->getEnvelopeInternal(), visitor);

            std::vector<float> vec(vector_dimension, 0.0f);
            for (size_t idx : visitor.found_indices) {
                if (prepared_poly->intersects(grid_cell_polygons[idx].get())) {
                    std::unique_ptr<geos::geom::Geometry> intersection(scaled_poly->intersection(grid_cell_polygons[idx].get()));
                    if (intersection && !intersection->isEmpty()) {
                        vec[idx] = intersection->getArea() / scaled_poly->getArea();
                    }
                }
            }

            out_file << data.geo_id << "\t";
            for (size_t i = 0; i < vec.size(); ++i) {
                out_file << std::fixed << std::setprecision(6) << vec[i] << (i == vec.size() - 1 ? "" : " ");
            }
            out_file << "\n";
            database_vectors_written++;

            if (database_vectors_written % 5000 == 0) {
                std::cout << "Processed " << database_vectors_written << " polygons..." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred during vectorization: " << e.what() << std::endl;
    }
    out_file.close();
    std::cout << "Finished processing. Total vectors written: " << database_vectors_written << std::endl;

    // --- 3. Finalize file header ---
    std::cout << "Updating vector file header..." << std::endl;
    std::fstream file_stream(output_file, std::ios::in | std::ios::out);
    if (file_stream.is_open()) {
        std::string existing_content((std::istreambuf_iterator<char>(file_stream)), std::istreambuf_iterator<char>());
        file_stream.seekp(0);
        file_stream << database_vectors_written << " " << vector_dimension << "\n";
        // Find where the old content started (after the first line)
        size_t first_line_end = existing_content.find('\n');
        if (first_line_end != std::string::npos) {
            file_stream << existing_content.substr(first_line_end + 1);
        }
        file_stream.close();
        std::cout << "Header updated successfully." << std::endl;
    } else {
        std::cerr << "Error: Could not re-open output file to update header." << std::endl;
    }

    std::cout << "\nEncoding Successful. Vectors written to " << output_file << std::endl;
    return 0;
}
