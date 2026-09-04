#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hnsw_index.h"

std::vector<float> ParseVector(const std::string& str) {
  std::vector<float> vec;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
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
  if (argc < 2) {
    std::cout << "{\"status\":\"ok\"}\n";
    return 0;
  }

  std::string query_str = argv[1];
  std::size_t top_k = (argc >= 3) ? std::stoul(argv[2]) : 5;

  std::vector<float> query = ParseVector(query_str);
  if (query.empty()) {
    std::cerr << "{\"error\":\"Empty query vector\"}\n";
    return 1;
  }

  std::ifstream f("data/actors.json");
  if (!f.is_open()) {
    f.open("../data/actors.json");
  }

  fig::core::HNSWIndex index(16);

  if (f.is_open()) {
    std::string line;
    while (std::getline(f, line)) {
      size_t id_pos = line.find("\"id\":");
      size_t name_pos = line.find("\"name\":");
      size_t emb_pos = line.find("\"embedding\":[");
      if (emb_pos == std::string::npos) {
        emb_pos = line.find("\"embedding\": [");
      }

      if (id_pos != std::string::npos && emb_pos != std::string::npos) {
        uint64_t id = 0;
        size_t comma = line.find(',', id_pos);
        if (comma != std::string::npos) {
          id = std::stoull(line.substr(id_pos + 5, comma - id_pos - 5));
        }

        std::string name = "";
        if (name_pos != std::string::npos) {
          size_t q1 = line.find('"', name_pos + 7);
          size_t q2 = line.find('"', q1 + 1);
          if (q1 != std::string::npos && q2 != std::string::npos) {
            name = line.substr(q1 + 1, q2 - q1 - 1);
          }
        }

        size_t bracket_start = line.find('[', emb_pos);
        size_t bracket_end = line.find(']', bracket_start);
        if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
          std::string vec_str = line.substr(bracket_start + 1, bracket_end - bracket_start - 1);
          std::vector<float> vec = ParseVector(vec_str);
          if (!vec.empty() && id > 0) {
            index.InsertProfile(id, name, vec);
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
