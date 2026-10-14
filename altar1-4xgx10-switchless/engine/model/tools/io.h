// The driver's file helpers: .npy (little-endian, C order; int32/int64/uint16/float32 read, int32/float32 written)
// and the reference's token files (ref/tokens/*.tokens.json: {"tokens": [...]}).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/check.h"

namespace glmrun {

inline void npy_write(const std::string& path, const char* descr, const std::vector<size_t>& shape, const void* data,
                      size_t elem) {
  std::string sh = "(";
  size_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : i + 1 < shape.size() ? ", " : "");
    n *= shape[i];
  }
  sh += ")";
  std::string hdr = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + sh + ", }";
  while ((10 + hdr.size() + 1) % 64) hdr += ' ';
  hdr += '\n';
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) die("cannot write %s", path.c_str());
  const uint16_t hl = (uint16_t)hdr.size();
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(hdr.data(), 1, hdr.size(), f);
  std::fwrite(data, elem, n, f);
  std::fclose(f);
}

// reads a 1-D or 2-D integer .npy as int64 values
inline std::vector<int64_t> npy_read_ints(const std::string& path, std::vector<size_t>* shape = nullptr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot read %s", path.c_str());
  char magic[8];
  f.read(magic, 8);
  uint16_t hl = 0;
  if (magic[6] == 1) f.read(reinterpret_cast<char*>(&hl), 2);
  else {
    uint32_t h4 = 0;
    f.read(reinterpret_cast<char*>(&h4), 4);
    hl = (uint16_t)h4;
  }
  std::string hdr(hl, ' ');
  f.read(hdr.data(), hl);
  const size_t d = hdr.find("'descr': '") + 10;
  const std::string descr = hdr.substr(d, hdr.find('\'', d) - d);
  const size_t s0 = hdr.find('(') + 1, s1 = hdr.find(')');
  std::vector<size_t> shp;
  std::stringstream ss(hdr.substr(s0, s1 - s0));
  std::string tok;
  size_t n = 1;
  while (std::getline(ss, tok, ','))
    if (tok.find_first_of("0123456789") != std::string::npos) shp.push_back(std::stoull(tok)), n *= shp.back();
  if (shape) *shape = shp;
  std::vector<int64_t> out(n);
  if (descr == "<i4") {
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    for (size_t i = 0; i < n; ++i) out[i] = v[i];
  } else if (descr == "<i8") {
    f.read(reinterpret_cast<char*>(out.data()), n * 8);
  } else if (descr == "<u2") {
    std::vector<uint16_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 2);
    for (size_t i = 0; i < n; ++i) out[i] = v[i];
  } else {
    die("%s: dtype %s (integers only)", path.c_str(), descr.c_str());
  }
  return out;
}

// token ids from a .npy or a {"tokens": [...]} JSON file
inline std::vector<int32_t> read_tokens(const std::string& path) {
  std::vector<int32_t> t;
  if (path.size() > 4 && path.substr(path.size() - 4) == ".npy") {
    for (int64_t v : npy_read_ints(path)) t.push_back((int32_t)v);
    return t;
  }
  std::ifstream f(path);
  if (!f) die("cannot read %s", path.c_str());
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string s = ss.str();
  size_t p = s.find("\"tokens\"");
  if (p == std::string::npos) die("%s: no \"tokens\"", path.c_str());
  p = s.find('[', p);
  const size_t e = s.find(']', p);
  std::stringstream ls(s.substr(p + 1, e - p - 1));
  std::string tok;
  while (std::getline(ls, tok, ','))
    if (tok.find_first_of("0123456789") != std::string::npos) t.push_back(std::stoi(tok));
  return t;
}

}  // namespace glmrun
