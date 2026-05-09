#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <numbers>
#include "logger.h"

namespace vwt::benchmark {

class ModelSelector {
public:
    static std::vector<std::string> getModels() {
        std::vector<std::string> models;
        std::string modelDir = "benchmark/models/";

        if (std::filesystem::exists(modelDir) && std::filesystem::is_directory(modelDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(modelDir)) {
                if (entry.path().extension() == ".stl" || 
                    entry.path().extension() == ".obj" ||
                    entry.path().extension() == ".glb" ||
                    entry.path().extension() == ".gltf" ||
                    entry.path().extension() == ".fbx") {
                    models.push_back(entry.path().string());
                }
            }
        }

        if (models.empty()) {
            vwt::Logger::log("No models found in " + modelDir + ". Generating procedural sphere.");
            models.push_back(generateProceduralSphere());
        }

        return models;
    }

    static void cleanupProceduralModels() {
        std::string path = "benchmark/models/.procedural_sphere.stl";
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
    }

private:
    static std::string generateProceduralSphere() {
        std::string dir = "benchmark/models/";
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        std::string path = dir + ".procedural_sphere.stl";
        std::ofstream file(path);
        if (!file.is_open()) {
            vwt::Logger::log("[ERROR] Could not create procedural sphere file");
            return "";
        }

        const int latDivs = 32;
        const int lonDivs = 64;
        const float radius = 0.3f;
        const float center = 0.5f;

        file << "solid procedural_sphere\n";

        auto getPoint = [&](int lat, int lon) {
            float phi = (float)lat * std::numbers::pi_v<float> / (float)latDivs;
            float theta = (float)lon * 2.0f * std::numbers::pi_v<float> / (float)lonDivs;
            float x = center + radius * std::sin(phi) * std::cos(theta);
            float y = center + radius * std::sin(phi) * std::sin(theta);
            float z = center + radius * std::cos(phi);
            return std::vector<float>{x, y, z};
        };

        for (int lat = 0; lat < latDivs; ++lat) {
            for (int lon = 0; lon < lonDivs; ++lon) {
                auto p1 = getPoint(lat, lon);
                auto p2 = getPoint(lat + 1, lon);
                auto p3 = getPoint(lat, lon + 1);
                auto p4 = getPoint(lat + 1, lon + 1);

                // Triangle 1
                file << "  facet normal 0 0 0\n    outer loop\n";
                file << "      vertex " << p1[0] << " " << p1[1] << " " << p1[2] << "\n";
                file << "      vertex " << p2[0] << " " << p2[1] << " " << p2[2] << "\n";
                file << "      vertex " << p3[0] << " " << p3[1] << " " << p3[2] << "\n";
                file << "    endloop\n  endfacet\n";

                // Triangle 2
                file << "  facet normal 0 0 0\n    outer loop\n";
                file << "      vertex " << p2[0] << " " << p2[1] << " " << p2[2] << "\n";
                file << "      vertex " << p4[0] << " " << p4[1] << " " << p4[2] << "\n";
                file << "      vertex " << p3[0] << " " << p3[1] << " " << p3[2] << "\n";
                file << "    endloop\n  endfacet\n";
            }
        }

        file << "endsolid procedural_sphere\n";
        file.close();

        return path;
    }
};

} // namespace vwt::benchmark
