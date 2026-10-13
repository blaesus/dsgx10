// The hot-row cache's index: CLOCK replacement over a fixed number of slots, found through an
// open-addressing table (linear probing, backward-shift deletion). New rows enter unreferenced, so
// rows seen once leave before rows seen again. Shared by the store and the hit-rate simulator.
#pragma once
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/check.h"

namespace engram {

constexpr uint32_t kNoSlot = 0xffffffffu;
constexpr uint64_t kNoKey = ~uint64_t(0);

inline uint64_t mix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return k;
}

inline void* big_alloc(size_t bytes) {
  if (!bytes) return nullptr;
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) die("engram: mmap %zu bytes: %s", bytes, strerror(errno));
  madvise(p, bytes, MADV_HUGEPAGE);
  return p;
}

// Slot indices in an open-addressing table; the keys live with the caller (key_of(slot)).
struct SlotTable {
  std::vector<uint32_t> t;
  size_t mask = 0;
  void init(size_t entries) {
    size_t n = 1;
    while (n < entries) n <<= 1;
    t.assign(n, kNoSlot);
    mask = n - 1;
  }
  void clear() { std::fill(t.begin(), t.end(), kNoSlot); }
  template <class KeyOf>
  uint32_t find(uint64_t key, KeyOf key_of) const {
    for (size_t h = mix64(key) & mask;; h = (h + 1) & mask) {
      const uint32_t s = t[h];
      if (s == kNoSlot || key_of(s) == key) return s;
    }
  }
  void insert(uint64_t key, uint32_t s) {
    size_t h = mix64(key) & mask;
    while (t[h] != kNoSlot) h = (h + 1) & mask;
    t[h] = s;
  }
  template <class KeyOf>
  void erase(uint64_t key, KeyOf key_of) {
    size_t i = mix64(key) & mask;
    while (key_of(t[i]) != key) i = (i + 1) & mask;
    for (size_t j = i;;) {
      j = (j + 1) & mask;
      if (t[j] == kNoSlot) break;
      const size_t k = mix64(key_of(t[j])) & mask;  // home of the entry at j
      const bool stays = i <= j ? (i < k && k <= j) : (i < k || k <= j);
      if (!stays) {
        t[i] = t[j];
        i = j;
      }
    }
    t[i] = kNoSlot;
  }
};

class ClockIndex {
 public:
  explicit ClockIndex(size_t slots = 0) { init(slots); }
  ~ClockIndex() {
    if (cap_) {
      munmap(keys_, cap_ * 8);
      munmap(ref_, cap_);
    }
  }
  ClockIndex(const ClockIndex&) = delete;
  ClockIndex& operator=(const ClockIndex&) = delete;

  void init(size_t slots) {
    cap_ = slots;
    if (!cap_) return;
    keys_ = static_cast<uint64_t*>(big_alloc(cap_ * 8));
    ref_ = static_cast<uint8_t*>(big_alloc(cap_));
    table_.init(cap_ * 2);
    clear();
  }
  void clear() {
    for (size_t i = 0; i < cap_; i++) keys_[i] = kNoKey;
    if (cap_) std::memset(ref_, 0, cap_);
    table_.clear();
    hand_ = 0;
  }
  size_t slots() const { return cap_; }

  // The slot holding `key`, marked referenced, or kNoSlot.
  uint32_t find(uint64_t key) {
    if (!cap_) return kNoSlot;
    const uint32_t s = table_.find(key, [this](uint32_t i) { return keys_[i]; });
    if (s != kNoSlot) ref_[s] = 1;
    return s;
  }
  // A slot for a new key (evicting by CLOCK if full); `evicted` says whether a row was dropped.
  uint32_t insert(uint64_t key, bool* evicted) {
    *evicted = false;
    if (!cap_) return kNoSlot;
    uint32_t s;
    for (;;) {
      s = uint32_t(hand_);
      hand_ = hand_ + 1 == cap_ ? 0 : hand_ + 1;
      if (keys_[s] == kNoKey) break;
      if (ref_[s]) {
        ref_[s] = 0;
        continue;
      }
      table_.erase(keys_[s], [this](uint32_t i) { return keys_[i]; });
      *evicted = true;
      break;
    }
    keys_[s] = key;
    ref_[s] = 0;
    table_.insert(key, s);
    return s;
  }

 private:
  size_t cap_ = 0, hand_ = 0;
  uint64_t* keys_ = nullptr;
  uint8_t* ref_ = nullptr;
  SlotTable table_;
};

}  // namespace engram
