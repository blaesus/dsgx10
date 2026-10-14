// The ring as cabled (docs/ring.md; the DeepSeek engine's docs/ring.md has the cabling): which device,
// address and GID each rank uses towards each neighbour. None of it is compiled in: it comes from the fleet
// file, read at run time by comm::fleet() below. This configuration has no fleet file of its own yet: it
// reads the DeepSeek engine's (ds41f-4xgx10-switchless/fleet/homelab.conf, the same four boxes), which the
// engine's CMakeLists.txt copies beside the binaries as fleet.conf and run4.sh installs beside its copies. Every box uses the same device names: port 0 (f0) faces the next rank,
// port 1 (f1) the previous one; "domain" 0 is PCI domain 0 (rocep1s0f* here), domain 1 is PCI domain 2
// (roceP2p1s0f*). Each (port, domain) pair is its own subnet per link.
#pragma once
#include <string>

namespace comm {

constexpr int kRanks = 4;
constexpr int kDomains = 2;
constexpr int kSlowLink = 0;  // link i joins ranks i and i + 1; L0 (ranks 0 - 1) is 100G, the others 200G

struct RankAddr {
  std::string name;                // RANK<r>_NAME: the box's name in logs
  std::string hostname;            // RANK<r>_HOSTNAME: `hostname` on that box
  std::string ssh;                 // RANK<r>_SSH: how rank 0's scripts reach it (unused by the engine)
  std::string next_ip[kDomains];   // RANK<r>_NEXT_IP<d>: this rank's address on the link to rank+1
  std::string prev_ip[kDomains];   // RANK<r>_PREV_IP<d>: this rank's address on the link to rank-1
  // The DeepSeek engine's per-rank keys, read (unknown keys are errors) and unused here: RANK<r>_WEIGHTS,
  // _HASHMAP, _ENGRAM, _SPILL as written (fleet_path expands them: {rank}, ~/), _ENGRAM_READ_BATCH (1 when
  // absent). This engine's own keys come with its own fleet file.
  std::string weights, hashmap, engram, spill;
  int engram_read_batch = 1;
};

// rank r's next_ip[d] and rank r+1's prev_ip[d] share a subnet.
struct Fleet {
  std::string file;                // where it was read from
  int gid_index = -1;              // GID_INDEX
  std::string next_dev[kDomains];  // NEXT_DEV<d>: the device facing rank+1, on every box
  std::string prev_dev[kDomains];  // PREV_DEV<d>: the device facing rank-1
  RankAddr rank[kRanks];
};

// The fleet file at `path` ("": fleet.conf beside this executable, where run4.sh and service.sh install
// it), read and checked on first use; dies with the file and line of anything it does not understand.
const Fleet& fleet(const std::string& path = "");
// A path value of the fleet or of the command line, for this box: every {rank} is `rank` (when >= 0),
// and a leading ~/ is $HOME.
std::string fleet_path(const std::string& value, int rank = -1);

}  // namespace comm
