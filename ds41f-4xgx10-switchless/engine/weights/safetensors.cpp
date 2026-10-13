#include "weights/safetensors.h"

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "weights/io.h"

namespace dsw {

namespace {

struct Parser {
  const std::string& s;
  size_t p = 0;

  [[noreturn]] void fail(const char* what) { die("json: %s at offset %zu", what, p); }
  void ws() {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
  }
  char peek() {
    ws();
    if (p >= s.size()) fail("unexpected end");
    return s[p];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    p++;
  }
  std::string str() {
    expect('"');
    std::string out;
    while (true) {
      if (p >= s.size()) fail("unterminated string");
      char c = s[p++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p >= s.size()) fail("bad escape");
      char e = s[p++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (p + 4 > s.size()) fail("bad \\u escape");
          unsigned cp = std::stoul(s.substr(p, 4), nullptr, 16);
          p += 4;
          // tensor names are ASCII; encode anything else as UTF-8 without surrogate pairing
          if (cp < 0x80) {
            out += char(cp);
          } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
          } else {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
          }
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }
  Json value() {
    Json v;
    char c = peek();
    if (c == '{') {
      p++;
      v.t = Json::T::Obj;
      if (peek() == '}') {
        p++;
        return v;
      }
      while (true) {
        std::string k = str();
        expect(':');
        v.obj.emplace_back(std::move(k), value());
        char d = peek();
        p++;
        if (d == '}') break;
        if (d != ',') fail("expected , or }");
      }
    } else if (c == '[') {
      p++;
      v.t = Json::T::Arr;
      if (peek() == ']') {
        p++;
        return v;
      }
      while (true) {
        v.arr.push_back(value());
        char d = peek();
        p++;
        if (d == ']') break;
        if (d != ',') fail("expected , or ]");
      }
    } else if (c == '"') {
      v.t = Json::T::Str;
      v.s = str();
    } else if (s.compare(p, 4, "true") == 0) {
      v.t = Json::T::Bool, v.b = true, p += 4;
    } else if (s.compare(p, 5, "false") == 0) {
      v.t = Json::T::Bool, v.b = false, p += 5;
    } else if (s.compare(p, 4, "null") == 0) {
      v.t = Json::T::Null, p += 4;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      size_t q = p;
      bool integral = true;
      if (s[q] == '-') q++;
      while (q < s.size() && std::strchr("0123456789.eE+-", s[q])) {
        if (s[q] == '.' || s[q] == 'e' || s[q] == 'E') integral = false;
        q++;
      }
      std::string num = s.substr(p, q - p);
      p = q;
      if (integral) {
        v.t = Json::T::Int;
        v.i = std::stoll(num);
      } else {
        v.t = Json::T::Num;
        v.d = std::stod(num);
      }
    } else {
      fail("unexpected token");
    }
    return v;
  }
};

std::string read_text(const std::string& path) {
  int fd = open_or_die(path, O_RDONLY);
  std::string out(file_size_of(fd), '\0');
  if (!out.empty()) pread_all(fd, out.data(), out.size(), 0);
  ::close(fd);
  return out;
}

}  // namespace

const Json* Json::get(const std::string& key) const {
  for (auto& [k, v] : obj)
    if (k == key) return &v;
  return nullptr;
}

Json Json::parse(const std::string& text) {
  Parser ps{text};
  Json v = ps.value();
  ps.ws();
  if (ps.p != text.size()) ps.fail("trailing data");
  return v;
}

uint32_t st_dtype_bytes(const std::string& t) {
  if (t == "F32" || t == "I32" || t == "U32") return 4;
  if (t == "BF16" || t == "F16" || t == "I16") return 2;
  if (t == "F8_E4M3" || t == "F8_E5M2" || t == "F8_E8M0" || t == "I8" || t == "U8" || t == "BOOL") return 1;
  if (t == "F64" || t == "I64" || t == "U64") return 8;
  die("unknown safetensors dtype %s", t.c_str());
}

const SrcTensor& Checkpoint::at(const std::string& name) const {
  auto it = tensors.find(name);
  if (it == tensors.end()) die("checkpoint has no tensor %s", name.c_str());
  return it->second;
}

Checkpoint Checkpoint::open(const std::string& dir) {
  Checkpoint ck;
  ck.dir = dir;
  DIR* d = ::opendir(dir.c_str());
  if (!d) die("opendir %s: %s", dir.c_str(), std::strerror(errno));
  while (dirent* e = ::readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) ck.shards.push_back(dir + "/" + n);
  }
  ::closedir(d);
  std::sort(ck.shards.begin(), ck.shards.end());
  if (ck.shards.empty()) die("no .safetensors files in %s", dir.c_str());

  for (int si = 0; si < (int)ck.shards.size(); si++) {
    int fd = open_or_die(ck.shards[si], O_RDONLY);
    uint64_t fsz = file_size_of(fd);
    uint64_t hlen = 0;
    pread_all(fd, &hlen, 8, 0);
    if (hlen == 0 || hlen > (100u << 20) || 8 + hlen > fsz) die("%s: bad header length", ck.shards[si].c_str());
    std::string htxt(hlen, '\0');
    pread_all(fd, htxt.data(), hlen, 8);
    ::close(fd);
    ck.shard_bytes.push_back(fsz);
    Json h = Json::parse(htxt);
    if (h.t != Json::T::Obj) die("%s: header is not an object", ck.shards[si].c_str());
    uint64_t base = 8 + hlen;
    for (auto& [name, v] : h.obj) {
      if (name == "__metadata__") continue;
      const Json* dt = v.get("dtype");
      const Json* sh = v.get("shape");
      const Json* off = v.get("data_offsets");
      if (!dt || !sh || !off || off->arr.size() != 2) die("%s: malformed entry %s", ck.shards[si].c_str(), name.c_str());
      SrcTensor t;
      t.name = name;
      t.shard = si;
      t.dtype = dt->s;
      uint64_t n = 1;
      for (auto& x : sh->arr) {
        t.shape.push_back((uint64_t)x.i);
        n *= (uint64_t)x.i;
      }
      t.offset = base + (uint64_t)off->arr[0].i;
      t.bytes = (uint64_t)(off->arr[1].i - off->arr[0].i);
      if (t.bytes != n * st_dtype_bytes(t.dtype)) die("%s: size mismatch for %s", ck.shards[si].c_str(), name.c_str());
      if (t.offset + t.bytes > fsz) die("%s: %s runs past the end of the file", ck.shards[si].c_str(), name.c_str());
      if (!ck.tensors.emplace(name, std::move(t)).second) die("tensor %s appears twice", name.c_str());
    }
  }

  // the index, when present, must name exactly the tensors found and put each in its shard
  std::string idx_path = dir + "/model.safetensors.index.json";
  if (::access(idx_path.c_str(), R_OK) == 0) {
    Json idx = Json::parse(read_text(idx_path));
    const Json* wm = idx.get("weight_map");
    if (!wm) die("%s has no weight_map", idx_path.c_str());
    if (wm->obj.size() != ck.tensors.size())
      die("index names %zu tensors, shards hold %zu", wm->obj.size(), ck.tensors.size());
    for (auto& [name, file] : wm->obj) {
      auto it = ck.tensors.find(name);
      if (it == ck.tensors.end()) die("index names %s, which no shard holds", name.c_str());
      const std::string& path = ck.shards[it->second.shard];
      if (path.size() < file.s.size() || path.compare(path.size() - file.s.size(), file.s.size(), file.s) != 0)
        die("index puts %s in %s, found in %s", name.c_str(), file.s.c_str(), path.c_str());
    }
  }
  return ck;
}

}  // namespace dsw
