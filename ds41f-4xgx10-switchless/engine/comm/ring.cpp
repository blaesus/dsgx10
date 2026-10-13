// Collectives for the four-box ring: bootstrap, RDMA setup and the proxy thread (see ring.h).
#include "comm/ring.h"

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "common/check.h"

#define CU_CHECK(x)                                                         \
  do {                                                                      \
    CUresult r_ = (x);                                                      \
    if (r_ != CUDA_SUCCESS) {                                               \
      const char* s_ = "?";                                                 \
      cuGetErrorString(r_, &s_);                                            \
      die("%s:%d: %s: %s", __FILE__, __LINE__, #x, s_);                     \
    }                                                                       \
  } while (0)

namespace comm {
namespace {

// Flags per buffer set, one cache line each. A/B: first and second half of a neighbour's direct write
// (A is the half this rank relays onward, so it is sent first).
enum FlagIdx { kFromPrevA, kFromPrevB, kFromNextA, kFromNextB, kOpp0, kOpp1, kPost, kFlagsPerParity = 8 };

// Byte offsets inside the region; identical on every rank, so remote address = peer base + offset.
struct Layout {
  size_t cap = 0, send[kSlots], from_prev[kSlots], from_next[kSlots], opp[kSlots], flags = 0, total = 0;
  size_t flag(int p, int k) const { return flags + (size_t)(p * kFlagsPerParity + k) * sizeof(Flag); }
};

Layout make_layout(size_t cap) {
  Layout L;
  L.cap = (cap + 4095) & ~size_t(4095);
  for (int p = 0; p < kSlots; p++) {
    L.send[p] = (0 * kSlots + p) * L.cap;
    L.from_prev[p] = (1 * kSlots + p) * L.cap;
    L.from_next[p] = (2 * kSlots + p) * L.cap;
    L.opp[p] = (3 * kSlots + p) * L.cap;
  }
  L.flags = 4 * kSlots * L.cap;
  L.total = L.flags + kSlots * kFlagsPerParity * sizeof(Flag);
  return L;
}

// The relayed half of an n-byte contribution starts at h; [0, h) goes one way, [h, n) the other.
inline uint32_t split_at(uint32_t n) { return (n / 2) & ~63u; }

// Reduce-scatter: the opposite rank's quarter (q bytes) goes [0, h) through prev and [h, q) through
// next, slow_pct percent of it on the path that crosses the slow link.
inline uint32_t rs_split(int sender, uint32_t q, int slow_pct) {
  const uint32_t part = (uint32_t)((uint64_t)q * slow_pct / 100) & ~63u;
  // through next the path takes links sender and sender + 1 (link i joins ranks i and i + 1)
  const bool next_slow = sender == kSlowLink || (sender + 1) % kRanks == kSlowLink;
  return next_slow ? q - part : part;
}

// A byte range of a contribution; it lands at the same offset in the receiver's buffer.
struct Seg {
  uint32_t off, len;
};

// What `sender`'s post of kind op (n bytes) sends to one neighbour: A, the part that neighbour
// relays on to the sender's opposite rank (sent first, so the relay can start early), and B, the
// rest. All-exchange: the whole contribution; next relays [h, n), prev relays [0, h).
// Reduce-scatter: quarter k (q = n / 4 bytes) is for rank k; a neighbour gets its own quarter (B)
// and relays its share of the opposite rank's (A, rs_split).
void direct_segs(uint32_t op, int sender, bool to_next, uint32_t n, int slow_pct, Seg& a, Seg& b) {
  if (op == kOpReduceScatter) {
    const uint32_t q = n / 4, hq = rs_split(sender, q, slow_pct), opp = (sender + 2) % kRanks * q;
    if (to_next) {
      a = {opp + hq, q - hq};
      b = {(sender + 1) % kRanks * q, q};
    } else {
      a = {opp, hq};
      b = {(sender + kRanks - 1) % kRanks * q, q};
    }
  } else {
    const uint32_t h = split_at(n);
    if (to_next) {
      a = {h, n - h};
      b = {0, h};
    } else {
      a = {0, h};
      b = {h, n - h};
    }
  }
}

// The part of a neighbour's post that `rank` relays: to prev, what next sent it as A; to next,
// what prev sent it as A.
Seg relay_seg(uint32_t op, int rank, bool to_prev, uint32_t n, int slow_pct) {
  Seg a, b;
  if (to_prev)
    direct_segs(op, (rank + 1) % kRanks, false, n, slow_pct, a, b);
  else
    direct_segs(op, (rank + kRanks - 1) % kRanks, true, n, slow_pct, a, b);
  return a;
}

struct PeerInfo {
  uint32_t qpn, psn, rkey, pad;
  uint64_t base;
  uint8_t gid[16];
};

// One RC queue pair to one neighbour on one PCI domain, with the region registered on its device.
struct Link {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  ibv_mtu mtu = IBV_MTU_4096;
  PeerInfo me{}, peer{};
  uint32_t max_inline = 0;
  int unsignaled = 0;
};

enum WrKind : uint64_t { kWrDirect = 0, kWrRelayToPrev = 1, kWrRelayToNext = 2 };
inline uint64_t wr_id(WrKind k, uint64_t seq) { return (uint64_t(k) << 56) | seq; }

// ---------------------------------------------------------------- TCP bootstrap

int tcp_listen(const char* ip, int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) die("bad address %s", ip);
  if (bind(s, (sockaddr*)&a, sizeof a)) die("bind %s:%d: %s", ip, port, strerror(errno));
  if (listen(s, 4)) die("listen: %s", strerror(errno));
  return s;
}

int tcp_connect(const char* ip, int port, int timeout_s) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) die("bad address %s", ip);
  auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (!connect(s, (sockaddr*)&a, sizeof a)) {
      int one = 1;
      setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
      return s;
    }
    close(s);
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeout_s))
      die("connect %s:%d: %s", ip, port, strerror(errno));
    usleep(100 * 1000);
  }
}

int tcp_accept(int ls, int timeout_s) {
  pollfd pf{ls, POLLIN, 0};
  int r = poll(&pf, 1, timeout_s * 1000);
  if (r == 0) die("comm: the previous rank did not connect within %d s", timeout_s);
  if (r < 0) die("poll: %s", strerror(errno));
  int c = accept(ls, nullptr, nullptr);
  if (c < 0) die("accept: %s", strerror(errno));
  int one = 1;
  setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  return c;
}

void send_all(int fd, const void* p, size_t n) {
  const char* c = (const char*)p;
  while (n) {
    ssize_t w = write(fd, c, n);
    if (w <= 0) die("socket write: %s", strerror(errno));
    c += w;
    n -= w;
  }
}

void recv_all(int fd, void* p, size_t n) {
  char* c = (char*)p;
  while (n) {
    ssize_t r = read(fd, c, n);
    if (r <= 0) die("socket read: %s", r ? strerror(errno) : "peer closed");
    c += r;
    n -= r;
  }
}

// ---------------------------------------------------------------- RDMA setup

// The GID as text: a RoCE v2 GID of an IPv4 address (::ffff:a.b.c.d) as that address.
std::string gid_text(const uint8_t* g) {
  static const uint8_t v4[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  char s[INET6_ADDRSTRLEN] = "?";
  if (!memcmp(g, v4, 12)) inet_ntop(AF_INET, g + 12, s, sizeof s);
  else inet_ntop(AF_INET6, g, s, sizeof s);
  return s;
}

// The link's device must carry the fleet's address for it (`key`=`ip`) at the fleet's GID index: a check
// that the fleet file describes this box, before anything waits on a neighbour.
void open_link(Link& L, const char* devname, int gid_index, const std::string& ip, const std::string& key,
               const std::string& fleet_file, void* region, size_t bytes, uint32_t psn) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  for (int i = 0; i < n && !L.ctx; i++)
    if (!strcmp(ibv_get_device_name(list[i]), devname)) L.ctx = ibv_open_device(list[i]);
  ibv_free_device_list(list);
  if (!L.ctx) die("RDMA device %s not found", devname);
  L.pd = ibv_alloc_pd(L.ctx);
  L.cq = ibv_create_cq(L.ctx, 4096, nullptr, nullptr, 0);
  if (!L.pd || !L.cq) die("%s: pd/cq", devname);
  ibv_qp_init_attr qia{};
  qia.send_cq = L.cq;
  qia.recv_cq = L.cq;
  qia.qp_type = IBV_QPT_RC;
  qia.sq_sig_all = 0;
  qia.cap.max_send_wr = 2048;
  qia.cap.max_recv_wr = 1;
  qia.cap.max_send_sge = 1;
  qia.cap.max_recv_sge = 1;
  qia.cap.max_inline_data = 256;
  L.qp = ibv_create_qp(L.pd, &qia);
  if (!L.qp) die("%s: create_qp: %s", devname, strerror(errno));
  L.max_inline = qia.cap.max_inline_data;
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_INIT;
  at.pkey_index = 0;
  at.port_num = 1;
  at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(L.qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    die("%s: qp init", devname);
  // No relaxed ordering: a flag written after its data must not become visible first.
  L.mr = ibv_reg_mr(L.pd, region, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!L.mr) die("%s: ibv_reg_mr(%zu bytes): %s", devname, bytes, strerror(errno));
  ibv_port_attr pa{};
  if (ibv_query_port(L.ctx, 1, &pa)) die("%s: query_port", devname);
  L.mtu = pa.active_mtu;
  ibv_gid gid;
  if (ibv_query_gid(L.ctx, 1, gid_index, &gid)) die("%s: query_gid %d", devname, gid_index);
  uint8_t want[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  if (inet_pton(AF_INET, ip.c_str(), want + 12) != 1)
    die("comm: %s: %s=%s is not an IPv4 address", fleet_file.c_str(), key.c_str(), ip.c_str());
  if (memcmp(gid.raw, want, 16)) {
    char host[256] = {0};
    gethostname(host, sizeof host - 1);
    die("comm: on %s, %s's GID %d is %s, but the fleet file %s has %s=%s: this box's link is not addressed as "
        "the fleet file says, or this is not that rank's box",
        host, devname, gid_index, gid_text(gid.raw).c_str(), fleet_file.c_str(), key.c_str(), ip.c_str());
  }
  L.me.qpn = L.qp->qp_num;
  L.me.psn = psn & 0xffffff;
  L.me.rkey = L.mr->rkey;
  L.me.base = (uint64_t)region;
  memcpy(L.me.gid, gid.raw, 16);
}

void connect_link(Link& L, const char* devname, int gid_index) {
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_RTR;
  at.path_mtu = L.mtu;
  at.dest_qp_num = L.peer.qpn;
  at.rq_psn = L.peer.psn;
  at.max_dest_rd_atomic = 1;
  at.min_rnr_timer = 12;
  at.ah_attr.is_global = 1;
  memcpy(at.ah_attr.grh.dgid.raw, L.peer.gid, 16);
  at.ah_attr.grh.sgid_index = gid_index;
  at.ah_attr.grh.hop_limit = 1;
  at.ah_attr.port_num = 1;
  if (ibv_modify_qp(L.qp, &at,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    die("%s: qp rtr: %s", devname, strerror(errno));
  memset(&at, 0, sizeof at);
  at.qp_state = IBV_QPS_RTS;
  at.timeout = 14;
  at.retry_cnt = 7;
  at.rnr_retry = 7;
  at.sq_psn = L.me.psn;
  at.max_rd_atomic = 1;
  if (ibv_modify_qp(L.qp, &at,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC))
    die("%s: qp rts: %s", devname, strerror(errno));
}

void close_link(Link& L) {
  if (L.qp) ibv_destroy_qp(L.qp);
  if (L.mr) ibv_dereg_mr(L.mr);
  if (L.cq) ibv_destroy_cq(L.cq);
  if (L.pd) ibv_dealloc_pd(L.pd);
  if (L.ctx) ibv_close_device(L.ctx);
  L = Link{};
}

// ---------------------------------------------------------------- host region

// cuMemCreate on the host NUMA node, mapped for the GPU and the CPU at one address: GPU reads run
// at full DRAM speed (cudaHostAlloc memory reads ~25 % slower, micro/hostmem.cu) and the NIC can
// register it.
struct HostRegion {
  void* ptr = nullptr;
  size_t size = 0;
  CUmemGenericAllocationHandle handle = 0;
};

HostRegion alloc_region(size_t bytes) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = 0;
  size_t gran = 0;
  CU_CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  HostRegion R;
  R.size = (bytes + gran - 1) / gran * gran;
  CU_CHECK(cuMemCreate(&R.handle, R.size, &prop, 0));
  CUdeviceptr va = 0;
  CU_CHECK(cuMemAddressReserve(&va, R.size, gran, 0, 0));
  CU_CHECK(cuMemMap(va, R.size, 0, R.handle, 0));
  CUmemAccessDesc acc[2]{};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc[0].location.id = dev;
  acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  acc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  acc[1].location.id = 0;
  acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(va, R.size, acc, 2));
  R.ptr = (void*)va;
  memset(R.ptr, 0, R.size);
  return R;
}

void free_region(HostRegion& R) {
  if (!R.ptr) return;
  cuMemUnmap((CUdeviceptr)R.ptr, R.size);
  cuMemAddressFree((CUdeviceptr)R.ptr, R.size);
  cuMemRelease(R.handle);
  R = HostRegion{};
}

inline uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return t.tv_sec * 1000000000ull + t.tv_nsec;
}

inline uint64_t load_acquire(const uint64_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline uint32_t load_relaxed(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
// Before handing data that another agent (GPU or NIC) wrote to the NIC: the flag load above is an
// acquire (LDAR); this also orders it against the doorbell for observers in the outer domain.
inline void io_fence() { asm volatile("dmb osh" ::: "memory"); }

}  // namespace

// ---------------------------------------------------------------- Ring

struct Ring::Impl {
  Options opt;
  const Fleet* F = nullptr;  // the fleet file (ring mode, or to find the rank)
  int rank = 0, next = 0, prev = 0;
  Layout L;
  HostRegion region;
  uint8_t* base = nullptr;
  Link to_next[kDomains], to_prev[kDomains];
  int sock_next = -1, sock_prev = -1, sock_listen = -1;
  DevChannel dev{};
  unsigned long long* d_seq = nullptr;
  unsigned int* d_counters = nullptr;
  std::thread proxy;
  std::atomic<bool> stop{false};
  // proxy state (proxy thread only, read racily by describe())
  uint64_t next_post = 1, next_relay_prev = 1, next_relay_next = 1;
  uint64_t done_relay_prev = 0, done_relay_next = 0;
  uint64_t held_prev = 0, held_next = 0;  // last relay counted as held back
  ProxyStats st{};
  std::vector<TraceRec> tr;
  uint64_t next_arrival = 1;

  Flag* flag(int p, int k) { return (Flag*)(base + L.flag(p, k)); }

  void bootstrap();
  void proxy_loop();
  bool post_chain(Link& l, ibv_send_wr* wrs, int n);
  int poll(Link& l);
  void post_direct(uint64_t seq, uint32_t n, uint32_t op);
  void post_relay(bool to_prev, uint64_t seq, uint32_t n, uint32_t op);
  void loopback_loop();
};

int rank_from_hostname(const std::string& fleet_file) {
  char h[256] = {0};
  gethostname(h, sizeof h - 1);
  const Fleet& F = fleet(fleet_file);
  for (int r = 0; r < kRanks; r++)
    if (!F.rank[r].hostname.empty() && F.rank[r].hostname == h) return r;
  return -1;
}

Ring::Ring(const Options& o) : impl_(new Impl) {
  Impl& I = *impl_;
  I.opt = o;
  const bool net = o.mode == Options::kRing;
  if (net || o.rank < 0) I.F = &fleet(o.fleet);
  if (o.rank >= kRanks) die("comm: rank %d, but the ring has %d", o.rank, kRanks);
  I.rank = o.rank >= 0 ? o.rank : rank_from_hostname(o.fleet);
  if (I.rank < 0) {
    char h[256] = {0};
    gethostname(h, sizeof h - 1);
    die("comm: this host, %s, is no RANK<r>_HOSTNAME in the fleet file %s (or pass --rank)", h, I.F->file.c_str());
  }
  I.next = (I.rank + 1) % kRanks;
  I.prev = (I.rank + kRanks - 1) % kRanks;
  CUDA_CHECK(cudaFree(nullptr));  // create the primary context the driver calls below use
  CU_CHECK(cuInit(0));
  I.L = make_layout(o.cap);
  I.region = alloc_region(I.L.total);
  I.base = (uint8_t*)I.region.ptr;
  if (net) {
    const Fleet& F = *I.F;
    const RankAddr& me = F.rank[I.rank];
    const std::string key = "RANK" + std::to_string(I.rank) + "_";
    std::random_device rd;
    for (int d = 0; d < kDomains; d++) {
      const std::string dd = std::to_string(d);
      open_link(I.to_next[d], F.next_dev[d].c_str(), F.gid_index, me.next_ip[d], key + "NEXT_IP" + dd, F.file,
                I.region.ptr, I.region.size, rd());
      open_link(I.to_prev[d], F.prev_dev[d].c_str(), F.gid_index, me.prev_ip[d], key + "PREV_IP" + dd, F.file,
                I.region.ptr, I.region.size, rd());
    }
    I.bootstrap();
    for (int d = 0; d < kDomains; d++) {
      connect_link(I.to_next[d], F.next_dev[d].c_str(), F.gid_index);
      connect_link(I.to_prev[d], F.prev_dev[d].c_str(), F.gid_index);
    }
  }
  CUDA_CHECK(cudaMalloc(&I.d_seq, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&I.d_counters, 2 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(I.d_seq, 0, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(I.d_counters, 0, 2 * sizeof(unsigned int)));

  DevChannel& D = I.dev;
  D.rank = I.rank;
  D.cap = (uint32_t)I.L.cap;
  for (int p = 0; p < kSlots; p++) {
    D.src[p][I.rank] = I.base + I.L.send[p];
    D.src[p][I.prev] = I.base + I.L.from_prev[p];
    D.src[p][I.next] = I.base + I.L.from_next[p];
    D.src[p][(I.rank + 2) % kRanks] = I.base + I.L.opp[p];
    D.wait[p][kWaitPrevA] = I.flag(p, kFromPrevA);
    D.wait[p][kWaitPrevB] = I.flag(p, kFromPrevB);
    D.wait[p][kWaitNextA] = I.flag(p, kFromNextA);
    D.wait[p][kWaitNextB] = I.flag(p, kFromNextB);
    D.wait[p][kWaitOpp0] = I.flag(p, kOpp0);
    D.wait[p][kWaitOpp1] = I.flag(p, kOpp1);
    D.post[p] = I.flag(p, kPost);
  }
  D.seq = I.d_seq;
  D.ready = I.d_seq + 1;
  D.arrive = I.d_counters;
  D.depart = I.d_counters + 1;

  I.tr.assign(o.trace > 0 ? o.trace + 2 : 0, TraceRec{});
  barrier();  // every QP is ready to receive before anyone sends
  if (o.mode != Options::kLoopNone) I.proxy = std::thread([&I] {
    if (I.opt.proxy_cpu >= 0) {
      cpu_set_t cs;
      CPU_ZERO(&cs);
      CPU_SET(I.opt.proxy_cpu, &cs);
      pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    if (I.opt.mode == Options::kRing) I.proxy_loop();
    else I.loopback_loop();
  });
  barrier();
  if (o.verbose)
    std::fprintf(stderr, "comm: rank %d (%s) up, region %zu bytes, cap %zu, mtu %d\n", I.rank,
                 I.F ? I.F->rank[I.rank].name.c_str() : "loop", I.region.size, I.L.cap,
                 128 << I.to_next[0].mtu);
}

Ring::~Ring() {
  Impl& I = *impl_;
  barrier();
  I.stop = true;
  if (I.proxy.joinable()) I.proxy.join();
  barrier();
  for (int d = 0; d < kDomains; d++) {
    close_link(I.to_next[d]);
    close_link(I.to_prev[d]);
  }
  free_region(I.region);
  cudaFree(I.d_seq);
  cudaFree(I.d_counters);
  if (I.sock_next >= 0) close(I.sock_next);
  if (I.sock_prev >= 0) close(I.sock_prev);
  if (I.sock_listen >= 0) close(I.sock_listen);
}

int Ring::rank() const { return impl_->rank; }
const DevChannel& Ring::dev() const { return impl_->dev; }
ProxyStats Ring::stats() const { return impl_->st; }
const TraceRec* Ring::trace() const { return impl_->tr.empty() ? nullptr : impl_->tr.data(); }

// Rank r listens on its address towards r-1 and connects to r+1's address towards r: both on the
// cable between them. The two sockets then carry the QP details for both domains.
void Ring::Impl::bootstrap() {
  sock_listen = tcp_listen(F->rank[rank].prev_ip[0].c_str(), opt.port);
  sock_next = tcp_connect(F->rank[next].prev_ip[0].c_str(), opt.port, opt.bootstrap_timeout_s);
  sock_prev = tcp_accept(sock_listen, opt.bootstrap_timeout_s);
  PeerInfo mine_next[kDomains], mine_prev[kDomains], theirs_next[kDomains], theirs_prev[kDomains];
  for (int d = 0; d < kDomains; d++) {
    mine_next[d] = to_next[d].me;
    mine_prev[d] = to_prev[d].me;
  }
  int32_t hello[2] = {rank, (int32_t)L.cap}, got_n[2], got_p[2];
  send_all(sock_next, hello, sizeof hello);
  send_all(sock_prev, hello, sizeof hello);
  recv_all(sock_next, got_n, sizeof got_n);
  recv_all(sock_prev, got_p, sizeof got_p);
  if (got_n[0] != next || got_p[0] != prev) die("comm: ring order mismatch (%d, %d)", got_n[0], got_p[0]);
  if (got_n[1] != (int32_t)L.cap || got_p[1] != (int32_t)L.cap) die("comm: ranks disagree on cap");
  send_all(sock_next, mine_next, sizeof mine_next);
  send_all(sock_prev, mine_prev, sizeof mine_prev);
  recv_all(sock_next, theirs_next, sizeof theirs_next);
  recv_all(sock_prev, theirs_prev, sizeof theirs_prev);
  for (int d = 0; d < kDomains; d++) {
    to_next[d].peer = theirs_next[d];
    to_prev[d].peer = theirs_prev[d];
  }
}

// Token twice round the ring: the first lap proves everyone arrived, the second releases them.
void Ring::barrier() {
  Impl& I = *impl_;
  if (I.opt.mode != Options::kRing) return;
  char t = 0;
  for (int lap = 0; lap < 2; lap++) {
    if (I.rank == 0) {
      send_all(I.sock_next, &t, 1);
      recv_all(I.sock_prev, &t, 1);
    } else {
      recv_all(I.sock_prev, &t, 1);
      send_all(I.sock_next, &t, 1);
    }
  }
}

void Ring::allgather_host(const void* mine, void* all, size_t n) {
  Impl& I = *impl_;
  if (I.opt.mode != Options::kRing) {
    for (int k = 0; k < kRanks; k++) memcpy((char*)all + k * n, mine, n);
    return;
  }
  if (n > 65536) die("allgather_host: %zu bytes is too large for the bootstrap sockets", n);
  char* a = (char*)all;
  memcpy(a + I.rank * n, mine, n);
  for (int step = 0; step < kRanks - 1; step++) {
    int out = (I.rank - step + kRanks) % kRanks, in = (I.rank - step - 1 + kRanks) % kRanks;
    send_all(I.sock_next, a + out * n, n);
    recv_all(I.sock_prev, a + in * n, n);
  }
}

std::string Ring::describe() const {
  Impl& I = *impl_;
  char b[1024];
  int o = snprintf(b, sizeof b,
                   "rank %d proxy: next_post %lu relay(prev %lu next %lu) done(prev %lu next %lu) posted %lu "
                   "relays %lu/%lu waits %lu retries %lu\n",
                   I.rank, I.next_post, I.next_relay_prev, I.next_relay_next, I.done_relay_prev,
                   I.done_relay_next, I.st.posted, I.st.relayed[0], I.st.relayed[1], I.st.relay_waits,
                   I.st.post_retries);
  static const char* names[] = {"fromPrevA", "fromPrevB", "fromNextA", "fromNextB", "opp0", "opp1", "post"};
  for (int p = 0; p < kSlots && o < (int)sizeof b; p++) {
    o += snprintf(b + o, sizeof b - o, "  slot %d:", p);
    for (int k = 0; k <= kPost && o < (int)sizeof b; k++)
      o += snprintf(b + o, sizeof b - o, " %s=%lu", names[k], (unsigned long)I.flag(p, k)->seq);
    o += snprintf(b + o, sizeof b - o, "\n");
  }
  return b;
}

// ---------------------------------------------------------------- proxy

int Ring::Impl::poll(Link& l) {
  ibv_wc wc[16];
  int n = ibv_poll_cq(l.cq, 16, wc);
  if (n < 0) die("comm: poll_cq failed");
  for (int i = 0; i < n; i++) {
    if (wc[i].status != IBV_WC_SUCCESS)
      die("comm: rank %d work completion error %s (wr_id %lx)", rank, ibv_wc_status_str(wc[i].status),
          (unsigned long)wc[i].wr_id);
    uint64_t kind = wc[i].wr_id >> 56, seq = wc[i].wr_id & ((1ull << 56) - 1);
    if (kind == kWrRelayToPrev) done_relay_prev = seq;
    if (kind == kWrRelayToNext) done_relay_next = seq;
  }
  return n;
}

// Posts a chain of work requests; signals its last one if the queue has gone 256 without a
// completion (or the caller already asked for one).
bool Ring::Impl::post_chain(Link& l, ibv_send_wr* wrs, int n) {
  for (int i = 0; i + 1 < n; i++) wrs[i].next = &wrs[i + 1];
  wrs[n - 1].next = nullptr;
  l.unsignaled += n;
  if (wrs[n - 1].send_flags & IBV_SEND_SIGNALED) {
    l.unsignaled = 0;
  } else if (l.unsignaled >= 256) {
    wrs[n - 1].send_flags |= IBV_SEND_SIGNALED;
    l.unsignaled = 0;
  }
  ibv_send_wr* bad = nullptr;
  for (;;) {
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    int e = ibv_post_send(l.qp, wrs, &bad);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    st.post_calls++;
    st.post_ns += (t1.tv_sec - t0.tv_sec) * 1000000000ull + (t1.tv_nsec - t0.tv_nsec);
    if (!e) return true;
    if (e != ENOMEM) die("comm: post_send: %s", strerror(e));
    st.post_retries++;
    poll(l);
    wrs = bad;  // resume from the first request that was not taken
  }
}

namespace {
struct Chain {
  ibv_send_wr wr[4];
  ibv_sge sg[4];
  Flag flag[2];  // inline sources: copied into the request at post time
  int n = 0;
  void write(const Link& l, const uint8_t* local, uint32_t len, uint64_t remote_off) {
    if (!len) return;
    ibv_send_wr& w = wr[n];
    ibv_sge& s = sg[n];
    w = ibv_send_wr{};
    s.addr = (uint64_t)local;
    s.length = len;
    s.lkey = l.mr->lkey;
    w.opcode = IBV_WR_RDMA_WRITE;
    w.sg_list = &s;
    w.num_sge = 1;
    w.send_flags = len <= l.max_inline ? IBV_SEND_INLINE : 0;
    w.wr.rdma.remote_addr = l.peer.base + remote_off;
    w.wr.rdma.rkey = l.peer.rkey;
    n++;
  }
  void set_flag(const Link& l, int slot, uint64_t seq, uint32_t nbytes, uint32_t op, uint64_t remote_off) {
    flag[slot].seq = seq;
    flag[slot].nbytes = nbytes;
    flag[slot].op = op;
    ibv_send_wr& w = wr[n];
    ibv_sge& s = sg[n];
    w = ibv_send_wr{};
    s.addr = (uint64_t)&flag[slot];
    s.length = 16;
    s.lkey = 0;
    w.opcode = IBV_WR_RDMA_WRITE;
    w.sg_list = &s;
    w.num_sge = 1;
    w.send_flags = IBV_SEND_INLINE;
    w.wr.rdma.remote_addr = l.peer.base + remote_off;
    w.wr.rdma.rkey = l.peer.rkey;
    n++;
  }
};
}  // namespace

// This rank's contribution to collective seq goes to both neighbours (direct_segs). Each neighbour
// relays part A onward to the opposite rank, so A is sent first, with its own flag. Large
// contributions send A on domain 0 and B on domain 1 at the same time; consumers wait for both.
void Ring::Impl::post_direct(uint64_t seq, uint32_t n, uint32_t op) {
  const int p = seq % kSlots;
  const uint8_t* mine = base + L.send[p];
  const bool split = n >= opt.split_domains_from;
  struct Side {
    Link* l[2];
    size_t buf, fa, fb;
    bool to_next;
  };
  Side sides[2] = {
      {{&to_next[0], &to_next[split ? 1 : 0]}, L.from_prev[p], L.flag(p, kFromPrevA), L.flag(p, kFromPrevB), true},
      {{&to_prev[0], &to_prev[split ? 1 : 0]}, L.from_next[p], L.flag(p, kFromNextA), L.flag(p, kFromNextB), false},
  };
  for (Side& sd : sides) {
    Seg a, b;
    direct_segs(op, rank, sd.to_next, n, opt.rs_slow_pct, a, b);
    if (!split) {
      Chain c;
      c.write(*sd.l[0], mine + a.off, a.len, sd.buf + a.off);
      c.set_flag(*sd.l[0], 0, seq, n, op, sd.fa);
      c.write(*sd.l[0], mine + b.off, b.len, sd.buf + b.off);
      c.set_flag(*sd.l[0], 1, seq, n, op, sd.fb);
      post_chain(*sd.l[0], c.wr, c.n);
    } else {
      Chain ca, cb;
      ca.write(*sd.l[0], mine + a.off, a.len, sd.buf + a.off);
      ca.set_flag(*sd.l[0], 0, seq, n, op, sd.fa);
      cb.write(*sd.l[1], mine + b.off, b.len, sd.buf + b.off);
      cb.set_flag(*sd.l[1], 0, seq, n, op, sd.fb);
      post_chain(*sd.l[0], ca.wr, ca.n);
      post_chain(*sd.l[1], cb.wr, cb.n);
    }
  }
  st.posted++;
}

// Forward a neighbour's part A to the other neighbour, for whom it comes from the opposite rank:
// what next sent goes to prev as its opp flag 0, what prev sent goes to next as its opp flag 1.
// All-exchange relays travel on opt.relay_domain; reduce-scatter relays on the other domain, which
// then carries part A plus the relay while B has the first one to itself.
void Ring::Impl::post_relay(bool to_prev_side, uint64_t seq, uint32_t n, uint32_t op) {
  const int p = seq % kSlots;
  const Seg r = relay_seg(op, rank, to_prev_side, n, opt.rs_slow_pct);
  const int d = op == kOpReduceScatter ? 1 - opt.relay_domain : opt.relay_domain;
  Chain c;
  if (to_prev_side) {
    Link& l = to_prev[d];
    c.write(l, base + L.from_next[p] + r.off, r.len, L.opp[p] + r.off);
    c.set_flag(l, 0, seq, n, op, L.flag(p, kOpp0));
    c.wr[c.n - 1].send_flags |= IBV_SEND_SIGNALED;
    c.wr[c.n - 1].wr_id = wr_id(kWrRelayToPrev, seq);
    post_chain(l, c.wr, c.n);
    st.relayed[0]++;
  } else {
    Link& l = to_next[d];
    c.write(l, base + L.from_prev[p] + r.off, r.len, L.opp[p] + r.off);
    c.set_flag(l, 0, seq, n, op, L.flag(p, kOpp1));
    c.wr[c.n - 1].send_flags |= IBV_SEND_SIGNALED;
    c.wr[c.n - 1].wr_id = wr_id(kWrRelayToNext, seq);
    post_chain(l, c.wr, c.n);
    st.relayed[1]++;
  }
}

// Diagnostics: answer each GPU post by setting the six wait flags locally.
void Ring::Impl::loopback_loop() {
  while (!stop.load(std::memory_order_relaxed)) {
    Flag* f = flag(next_post % kSlots, kPost);
    if (load_acquire(&f->seq) == next_post) {
      const int p = next_post % kSlots;
      for (int k : {kFromPrevA, kFromPrevB, kFromNextA, kFromNextB, kOpp0, kOpp1}) __atomic_store_n(&flag(p, k)->seq, next_post, __ATOMIC_RELEASE);
      st.posted++;
      next_post++;
    }
  }
}

// One thread, busy-polling: the GPU's doorbell, the two neighbours' "part A arrived" flags, and the
// completion queues. A relay of collective s may only be posted once the relay of s - 1 in the
// other direction has completed. The neighbour that next overwrites the buffer relay s - 1 read
// from (with s - 1 + kSlots) must first consume s, which needs relay s: so by then this rank's NIC
// is done with that buffer. The thread spins while there is traffic; after opt.idle_backoff_ms
// without any, it polls every opt.idle_poll_us (checked every 1024 rounds, off the hot path).
void Ring::Impl::proxy_loop() {
  const uint64_t backoff_ns = (uint64_t)opt.idle_backoff_ms * 1000000ull;
  uint64_t last_work = now_ns();
  unsigned rounds = 0;
  bool worked = false;
  while (!stop.load(std::memory_order_relaxed)) {
    {
      Flag* f = flag(next_post % kSlots, kPost);
      if (load_acquire(&f->seq) == next_post) {
        io_fence();
        if (next_post < tr.size()) tr[next_post].post = now_ns();
        post_direct(next_post, load_relaxed(&f->nbytes), load_relaxed(&f->op));
        next_post++;
        worked = true;
      }
    }
    {
      Flag* f = flag(next_relay_prev % kSlots, kFromNextA);
      if (load_acquire(&f->seq) == next_relay_prev) {
        if (done_relay_next + 1 >= next_relay_prev) {
          io_fence();
          post_relay(true, next_relay_prev, load_relaxed(&f->nbytes), load_relaxed(&f->op));
          if (next_relay_prev < tr.size()) tr[next_relay_prev].relay[0] = now_ns();
          next_relay_prev++;
          worked = true;
        } else if (held_prev != next_relay_prev) {
          held_prev = next_relay_prev;
          st.relay_waits++;
        }
      }
    }
    {
      Flag* f = flag(next_relay_next % kSlots, kFromPrevA);
      if (load_acquire(&f->seq) == next_relay_next) {
        if (done_relay_prev + 1 >= next_relay_next) {
          io_fence();
          post_relay(false, next_relay_next, load_relaxed(&f->nbytes), load_relaxed(&f->op));
          if (next_relay_next < tr.size()) tr[next_relay_next].relay[1] = now_ns();
          next_relay_next++;
          worked = true;
        } else if (held_next != next_relay_next) {
          held_next = next_relay_next;
          st.relay_waits++;
        }
      }
    }
    for (int d = 0; d < kDomains; d++) {
      if (poll(to_next[d])) worked = true;
      if (poll(to_prev[d])) worked = true;
    }
    if (backoff_ns && ++rounds == 1024) {
      rounds = 0;
      const uint64_t t = now_ns();
      if (worked) {
        last_work = t;
        worked = false;
      } else if (t - last_work > backoff_ns) {
        usleep(opt.idle_poll_us);
        rounds = 1023;  // backed off: check after every round
      }
    }
    if (next_arrival < tr.size()) {
      const int p = next_arrival % kSlots;
      static const int ks[kNumWait] = {kFromPrevA, kFromPrevB, kFromNextA, kFromNextB, kOpp0, kOpp1};
      int got = 0;
      for (int k = 0; k < kNumWait; k++) {
        uint64_t& t = tr[next_arrival].arrive[k];
        if (!t && load_acquire(&flag(p, ks[k])->seq) == next_arrival) t = now_ns();
        got += t != 0;
      }
      if (got == kNumWait) next_arrival++;
    }
  }
}

}  // namespace comm
