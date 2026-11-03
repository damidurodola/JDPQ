#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// A simple struct to hold the data from a row in the CSV
struct PolygonData {
    long long geo_id;
    std::string wkt;
    std::string geo_tag;
};

// A basic CSV reader that handles quoted WKT strings.
// This is a simplified parser and might need to be made more robust.
class CSVReader {
public:
    CSVReader(const std::string& filename, char delimiter = ',')
        : file(filename), delimiter(delimiter) {
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        // Read header
        std::getline(file, current_line);
    }

    bool next_row(PolygonData& polygon) {
        if (!std::getline(file, current_line)) {
            return false;
        }

        std::stringstream ss(current_line);
        std::string part;
        std::vector<std::string> parts;

        // This simple logic assumes the WKT is the first field and may be quoted.
        if (current_line.front() == '"') {
            // Find the closing quote of the WKT string
            size_t wkt_end = current_line.find('"', 1);
            if (wkt_end != std::string::npos) {
                parts.push_back(current_line.substr(1, wkt_end - 1)); // The WKT
                // The rest of the line
                std::string rest = current_line.substr(wkt_end + 2); // +2 to skip quote and comma
                std::stringstream rest_ss(rest);
                std::string item;
                while(std::getline(rest_ss, item, delimiter)) {
                    parts.push_back(item);
                }
            }
        } else {
             while(std::getline(ss, part, delimiter)) {
                parts.push_back(part);
            }
        }

        if (parts.size() >= 2) {
            polygon.wkt = parts[0];
            try {
                polygon.geo_id = std::stoll(parts[1]);
                if (parts.size() > 2) {
                    polygon.geo_tag = parts[2];
                } else {
                    polygon.geo_tag = "";
                }
                return true;
            } catch (const std::invalid_argument& e) {
                // Could not convert geo_id, skip row
                return next_row(polygon);
            }
        }
        return false; // Malformed row
    }

private:
    std::ifstream file;
    char delimiter;
    std::string current_line;
};

#endif // UTILS_HPP
