// Engram hashing, a port of ref/inference/engram.py's NgramHashState.
#include <cstdio>
#include <cstring>

#include "common/check.h"
#include "engram/engram.h"

namespace engram {

namespace {

constexpr char kMagic[8] = {'D', 'S', 'G', 'B', 'E', 'N', 'G', '1'};
constexpr int64_t kBucketStart = 16'000'000;  // engram_vocab_size: every bucket is the next prime above it

bool is_prime(int64_t n) {
  if (n < 2) return false;
  for (int64_t d = 2; d * d <= n; d++)
    if (n % d == 0) return false;
  return true;
}

}  // namespace

HashMap HashMap::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("engram: cannot open %s", path.c_str());
  HashMap m;
  char magic[8];
  int64_t h[7];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, kMagic, 8) != 0) die("engram: %s: bad magic", path.c_str());
  if (std::fread(h, 8, 7, f) != 7) die("engram: %s: short header", path.c_str());
  if (h[0] != 1 || h[4] != kLayers || h[5] != kMaxNgram || h[6] != kHeads)
    die("engram: %s: version %lld, layout %lld/%lld/%lld not supported", path.c_str(), (long long)h[0],
        (long long)h[4], (long long)h[5], (long long)h[6]);
  m.n_tokens = h[1];
  m.vocab = h[2];
  m.pad = h[3];
  bool ok = std::fread(m.layer_ids, 8, kLayers, f) == kLayers && std::fread(m.num_embeddings, 8, kLayers, f) == kLayers &&
            std::fread(m.mult, 8, kLayers * kMaxNgram, f) == size_t(kLayers * kMaxNgram) &&
            std::fread(m.prime, 8, kLayers * kCols, f) == size_t(kLayers * kCols) &&
            std::fread(m.offset, 8, kLayers * kCols, f) == size_t(kLayers * kCols);
  m.token_map.resize(m.n_tokens);
  ok = ok && std::fread(m.token_map.data(), 4, m.n_tokens, f) == size_t(m.n_tokens);
  std::fclose(f);
  if (!ok) die("engram: %s: short file", path.c_str());

  // The primes are the next unused primes above kBucketStart - 1, drawn layer by layer, n-gram size
  // by n-gram size, head by head; the offsets are their running sum within a layer.
  int64_t p = kBucketStart - 1;
  for (int l = 0; l < kLayers; l++) {
    int64_t sum = 0;
    for (int o = 0; o < kOrders; o++)
      for (int hd = 0; hd < kHeads; hd++) {
        do p++;
        while (!is_prime(p));
        if (m.prime[l][o][hd] != p) die("engram: %s: prime [%d][%d][%d] is %lld, expected %lld", path.c_str(), l, o, hd,
                                        (long long)m.prime[l][o][hd], (long long)p);
        if (m.offset[l][o * kHeads + hd] != sum) die("engram: %s: offset mismatch", path.c_str());
        sum += p;
      }
    if (sum != m.num_embeddings[l]) die("engram: %s: layer %d buckets sum to %lld, table has %lld rows", path.c_str(), l,
                                        (long long)sum, (long long)m.num_embeddings[l]);
  }
  for (int l = 0; l < kLayers; l++)
    for (int s = 0; s < kMaxNgram; s++)
      if (m.mult[l][s] <= 0 || (m.mult[l][s] & 1) == 0 || m.mult[l][s] > INT64_MAX / m.vocab)
        die("engram: %s: bad multiplier", path.c_str());
  for (int32_t v : m.token_map)
    if (v < 0 || v >= m.vocab) die("engram: %s: token map entry out of range", path.c_str());
  if (m.pad < 0 || m.pad >= m.vocab) die("engram: %s: bad pad id", path.c_str());
  return m;
}

void hash_positions(const HashMap& m, const int32_t* tokens, const uint8_t* live, int64_t first, int n, uint32_t* out) {
  for (int i = 0; i < n; i++) {
    const int64_t pos = first + i;
    // the n-gram ending at pos: t[s] is the token s places back, or pad once look-back is blocked
    // by the sequence start or a dead position (and stays blocked further back)
    int64_t t[kMaxNgram];
    bool blocked = false;
    for (int s = 0; s < kMaxNgram; s++) {
      const int64_t src = pos - s < 0 ? 0 : pos - s;
      blocked = blocked || pos < s || (live && !live[src]);
      t[s] = blocked ? m.pad : m.token_map[tokens[src]];
    }
    for (int l = 0; l < kLayers; l++) {
      uint32_t* o = out + (size_t(i) * kLayers + l) * kCols;
      // XOR the multiplied ids one look-back at a time: after step k the value hashes the (k+1)-gram
      uint64_t rolling = uint64_t(t[0] * m.mult[l][0]);
      for (int k = 1; k < kMaxNgram; k++) {
        rolling ^= uint64_t(t[k] * m.mult[l][k]);
        for (int hd = 0; hd < kHeads; hd++) {
          const int c = (k - 1) * kHeads + hd;
          o[c] = uint32_t(rolling % uint64_t(m.prime[l][k - 1][hd]) + uint64_t(m.offset[l][c]));
        }
      }
    }
  }
}

Split Split::of(const HashMap& m) {
  Split s;
  for (int l = 0; l < kLayers; l++)
    for (int r = 0; r <= kWorld; r++) s.lo[l][r] = m.num_embeddings[l] * r / kWorld;
  return s;
}

void CellMap::build(const Split& s, int layer, const uint32_t* ids, int n) {
  n_pos = n;
  owner.resize(size_t(n) * kCols);
  index.resize(size_t(n) * kCols);
  std::memset(count, 0, sizeof(count));
  for (int p = 0; p < n; p++)
    for (int c = 0; c < kCols; c++) {
      const int o = s.owner(layer, ids[(size_t(p) * kLayers + layer) * kCols + c]);
      owner[size_t(p) * kCols + c] = uint8_t(o);
      index[size_t(p) * kCols + c] = count[o]++;
    }
}

size_t CellMap::seg_bytes() const {
  uint32_t most = 0;
  for (int r = 0; r < kWorld; r++) most = count[r] > most ? count[r] : most;
  return (size_t(most) * kRowBytes + 15) & ~size_t(15);
}

void CellMap::offsets(uint32_t* out) const {
  const size_t seg = seg_bytes();
  if (seg * kWorld > 0xffffffffu) die("engram: %d positions do not fit 32-bit offsets", n_pos);
  for (size_t i = 0; i < size_t(n_pos) * kCols; i++) out[i] = uint32_t(owner[i] * seg + size_t(index[i]) * kRowBytes);
}

}  // namespace engram
