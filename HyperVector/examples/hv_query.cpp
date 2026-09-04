// ============================================================================
// HyperVector Query CLI - Native C++ HNSW Vector Search
// ============================================================================
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hnsw_index.h"

// Simple parser for float vectors from comma/space separated strings
std::vector<float> ParseVector(const std::string& str) {
  std::vector<float> vec;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // trim whitespace
    size_t start = item.find_first_not_of(" \t\r\n[");
    size_t end = item.find_last_not_of(" \t\r\n]");
    if (start != std::string::npos && end != std::string::npos) {
      try {
        vec.push_back(std::stof(item.substr(start, end - start + 1)));
      } catch (...) {}
    }
  }
  return vec;
}

int main(int argc, char* argv[]) {
  // If run with --demo or no arguments, print usage
  if (argc < 2) {
    std::cout << "{\"status\":\"ok\",\"engine\":\"HyperVector C++20 HNSW with AVX2 SIMD\"}\n";
    return 0;
  }

  // Load embeddings from a simple format or test
  // argv[1]: query vector string (comma-separated 128 floats)
  // argv[2]: top_k (default 5)
  std::string query_str = argv[1];
  std::size_t top_k = (argc >= 3) ? std::stoul(argv[2]) : 5;

  std::vector<float> query = ParseVector(query_str);
  if (query.empty()) {
    std::cerr << "{\"error\":\"Empty or invalid query vector\"}\n";
    return 1;
  }

  // Read actor embeddings from data/actors.json or stdin
  std::ifstream f("data/actors.json");
  if (!f.is_open()) {
    f.open("../data/actors.json");
  }

  fig::core::HNSWIndex index(16);

  if (f.is_open()) {
    std::string line;
    uint64_t current_id = 0;
    std::string current_name = "";
    std::vector<float> current_vec;
    bool in_embedding = false;

    while (std::getline(f, line)) {
      if (line.find("\"id\":") != std::string::npos) {
        size_t colon = line.find(':');
        size_t comma = line.find(',', colon);
        if (colon != std::string::npos) {
          current_id = std::stoull(line.substr(colon + 1, comma - colon - 1));
        }
      } else if (line.find("\"name\":") != std::string::npos) {
        size_t first = line.find('"');
        first = line.find('"', first + 1);
        first = line.find('"', first + 1);
        size_t last = line.find('"', first + 1);
        if (first != std::string::npos && last != std::string::npos) {
          current_name = line.substr(first + 1, last - first - 1);
        }
      } else if (line.find("\"embedding\": [") != std::string::npos) {
        in_embedding = true;
        current_vec.clear();
      } else if (in_embedding) {
        if (line.find("]") != std::string::npos) {
          in_embedding = false;
          if (!current_vec.empty() && current_id > 0) {
            index.InsertProfile(current_id, current_name, current_vec);
          }
        } else {
          size_t start = line.find_first_not_of(" \t\r\n,");
          size_t end = line.find_last_not_of(" \t\r\n,");
          if (start != std::string::npos && end != std::string::npos) {
            try {
              current_vec.push_back(std::stof(line.substr(start, end - start + 1)));
            } catch (...) {}
          }
        }
      }
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<uint64_t> matches = index.SearchKNN(query, top_k);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  std::cout << "{\"latency_us\":" << elapsed_us << ",\"total_nodes\":" << index.Size()
            << ",\"max_level\":" << index.MaxLevel() << ",\"matches\":[";
  for (size_t i = 0; i < matches.size(); ++i) {
    if (i > 0) std::cout << ",";
    std::cout << matches[i];
  }
  std::cout << "]}\n";

  return 0;
}
