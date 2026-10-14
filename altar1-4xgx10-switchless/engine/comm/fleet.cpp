// The fleet file (ring_config.h): KEY=value lines and # comments, read as engine/server/fleet.sh reads
// them for the scripts. Unknown keys, repeated keys and anything else are errors, so that a typo cannot
// leave a value at a default.
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "comm/ring_config.h"
#include "common/check.h"

namespace comm {
namespace {

bool key_char(char c) { return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }
bool value_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         (c != '\0' && std::strchr("._-/~:,@+%{}", c) != nullptr);
}
bool blank(char c) { return c == ' ' || c == '\t'; }

// fleet.conf beside this executable: where run4.sh and service.sh install the fleet file
std::string default_file() {
  char buf[PATH_MAX];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) die("comm: /proc/self/exe: %s", std::strerror(errno));
  buf[n] = 0;
  std::string p(buf);
  return p.substr(0, p.rfind('/') + 1) + "fleet.conf";
}

Fleet read_fleet(const std::string& file) {
  std::ifstream in(file);
  if (!in)
    die("comm: cannot read the fleet file %s (--fleet FILE; engine/comm/run4.sh installs FLEET beside its "
        "binaries as fleet.conf; the default is ds41f-4xgx10-switchless/fleet/homelab.conf)",
        file.c_str());
  Fleet F;
  F.file = file;
  std::string gid, unused[4], read_batch[kRanks];
  std::map<std::string, std::string*> keys = {{"GID_INDEX", &gid},      {"BIN_DIR", &unused[0]},
                                              {"LOG_DIR", &unused[1]},  {"SERVICE_DIR", &unused[2]},
                                              {"RING_LOCK", &unused[3]}};
  for (int d = 0; d < kDomains; d++) {
    keys["NEXT_DEV" + std::to_string(d)] = &F.next_dev[d];
    keys["PREV_DEV" + std::to_string(d)] = &F.prev_dev[d];
  }
  for (int r = 0; r < kRanks; r++) {
    RankAddr& R = F.rank[r];
    const std::string p = "RANK" + std::to_string(r) + "_";
    keys[p + "NAME"] = &R.name;
    keys[p + "HOSTNAME"] = &R.hostname;
    keys[p + "SSH"] = &R.ssh;
    for (int d = 0; d < kDomains; d++) {
      keys[p + "NEXT_IP" + std::to_string(d)] = &R.next_ip[d];
      keys[p + "PREV_IP" + std::to_string(d)] = &R.prev_ip[d];
    }
    keys[p + "WEIGHTS"] = &R.weights;
    keys[p + "HASHMAP"] = &R.hashmap;
    keys[p + "ENGRAM"] = &R.engram;
    keys[p + "ENGRAM_READ_BATCH"] = &read_batch[r];
    keys[p + "SPILL"] = &R.spill;
  }
  std::map<std::string, int> seen;
  std::string line;
  for (int n = 1; std::getline(in, line); n++) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t i = 0;
    while (i < line.size() && blank(line[i])) i++;
    if (i == line.size() || line[i] == '#') continue;
    // KEY=value, then nothing, or blanks and an optional # comment
    size_t k = 0;
    while (k < line.size() && key_char(line[k])) k++;
    size_t v = k + 1, e = v;
    while (e < line.size() && value_char(line[e])) e++;
    size_t t = e;
    while (t < line.size() && blank(line[t])) t++;
    const bool ok = k > 0 && line[0] >= 'A' && line[0] <= 'Z' && k < line.size() && line[k] == '=' &&
                    (t == line.size() || (t > e && line[t] == '#'));
    if (!ok) die("comm: %s:%d: not KEY=value (a value has no spaces or quotes): %s", file.c_str(), n, line.c_str());
    const std::string key = line.substr(0, k);
    const auto it = keys.find(key);
    if (it == keys.end()) die("comm: %s:%d: unknown key %s", file.c_str(), n, key.c_str());
    if (seen.count(key)) die("comm: %s:%d: %s again (line %d)", file.c_str(), n, key.c_str(), seen[key]);
    seen[key] = n;
    *it->second = line.substr(v, e - v);
  }
  auto need = [&](const std::string& key, const std::string& val) {
    if (val.empty()) die("comm: %s: no %s", file.c_str(), key.c_str());
  };
  need("GID_INDEX", gid);
  char* end = nullptr;
  const long g = std::strtol(gid.c_str(), &end, 10);
  if (*end || g < 0 || g > 255) die("comm: %s: GID_INDEX=%s is not an index", file.c_str(), gid.c_str());
  F.gid_index = (int)g;
  for (int d = 0; d < kDomains; d++) {
    need("NEXT_DEV" + std::to_string(d), F.next_dev[d]);
    need("PREV_DEV" + std::to_string(d), F.prev_dev[d]);
  }
  for (int r = 0; r < kRanks; r++) {
    RankAddr& R = F.rank[r];
    const std::string p = "RANK" + std::to_string(r) + "_";
    for (int d = 0; d < kDomains; d++) {
      need(p + "NEXT_IP" + std::to_string(d), R.next_ip[d]);
      need(p + "PREV_IP" + std::to_string(d), R.prev_ip[d]);
    }
    if (R.name.empty()) R.name = "rank " + std::to_string(r);
    if (!read_batch[r].empty()) {
      const long b = std::strtol(read_batch[r].c_str(), &end, 10);
      if (*end || read_batch[r][0] == '-' || b < 1 || b > 128)
        die("comm: %s: %sENGRAM_READ_BATCH=%s is not a whole number from 1 to 128", file.c_str(), p.c_str(),
            read_batch[r].c_str());
      R.engram_read_batch = (int)b;
    }
  }
  return F;
}

}  // namespace

const Fleet& fleet(const std::string& path) {
  static std::mutex mu;
  static std::map<std::string, std::unique_ptr<Fleet>> read;
  std::lock_guard<std::mutex> lock(mu);
  const std::string file = path.empty() ? default_file() : path;
  std::unique_ptr<Fleet>& f = read[file];
  if (!f) f = std::make_unique<Fleet>(read_fleet(file));
  return *f;
}

std::string fleet_path(const std::string& value, int rank) {
  std::string v = value;
  if (rank >= 0)
    for (size_t at = 0; (at = v.find("{rank}", at)) != std::string::npos;) {
      const std::string r = std::to_string(rank);
      v.replace(at, 6, r);
      at += r.size();
    }
  if (v != "~" && v.rfind("~/", 0) != 0) return v;
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + v.substr(1);
}

}  // namespace comm
