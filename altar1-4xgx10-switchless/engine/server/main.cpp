// altar1_rank: one rank of the altar-1 engine as a service (docs/serving.md). Every rank runs it with the same
// arguments (run4.sh, service.sh); rank 0 serves the front-end (altarfront) on --socket, ranks 1-3 follow its commands.
//   altar1_rank [--fleet FILE] [--rank R] --mode serve [options]
// Weights: --weights PATH ("%d": the rank; "fleet", the default: the fleet file's RANK<r>_WEIGHTS) with
//   --fill-experts (routed-expert pools the file has not filled yet run synthetic: the text is meaningless, the
//   serving is real), or --synthetic (no file). --pools N, --seed S: the synthetic pools.
// Serving: --socket PATH (/tmp/dsgb10-altar1-engine.sock), --max-ctx N (262144: positions a session may reach),
//   --slots N (4: sessions a decode step, B2), --slice N (64: time-slicing's turns, B1), --chunk N (8192: rows a
//   prefill chunk), --splits N (16: sparse attention's split-K at every row count, so that a session's tokens do
//   not depend on the others in its steps), --name NAME (altar-1).
// The cache pool (paged, docs/serving.md): pages of --page N positions (1024); every rank offers its MemAvailable
//   (--pool-basis avail, the default: CUDA's allocations reclaim the page cache, which free device memory leaves
//   out; --pool-basis free: the smaller of free device memory and MemAvailable, glm53's rule) after the weights and
//   the prefill buffers, less --margin-gib (8 on rank
//   0, --worker-margin-gib 8 on the others: the fleet's memory rule keeps 8 GiB available on every box), and all
//   take the smallest offer, at most --kv-gib (default: no cap). The memory guard: after the weights and the prefill
//   buffers every rank reports its MemAvailable, and all ranks stop when one has less than --min-free-gib (8) plus
//   one page; the MemAvailable after the pool is logged too. A 256k session (262,144 positions) is 23.6 GiB a rank.
// The disk tier: --disk-gib N (32; 0: none) in --disk-dir (the fleet's RANK<r>_SPILL), files rank<r>[.tag].spill
//   (--disk-tag T); each rank offers its directory's free space less --disk-margin-gib (32), rank 0 at most
//   --disk-rank0-gib (0: rank 0's data disk holds its 82 GB rank file with ~35 GB left, so the tier is off unless
//   this is set; every rank holds the whole cache, so the tier is as large as the smallest offer), and all take the
//   smallest offer.
// Kept sessions: --keep-gib N (0): a planned stop (SIGTERM) keeps up to N GiB of sessions for the next start, none
//   unused for --keep-days (7).
// Drafts (speculative decoding with the MTP layer, docs/serving.md "Drafts"): on by default, up to --spec-drafts K
//   (3, at most 5) a session a step, as many as its gate's acceptance pays for under --spec-cost C1,DROW (ms: a
//   step's first row with draft pass 1, each row more with its pass; default 52.4,13.0: altar-1's fit, docs/serving.md
//   § 3.1); --no-spec: plain decoding (the
//   same tokens). The model's Options::drafts is K (0 with --no-spec: no MTP work at all). --spec-force N (tests):
//   N drafts every step whatever the gate says; --spec-oracle (tests): a request identical to an earlier one (prompt,
//   sampling, max_new) takes that one's tokens as its drafts, about one in four changed.
// --step-timeout S (120): the watchdog's limit for one step or chunk. --index-scratch-mib N (512): the prefill
// indexer's score scratch (128 with --chunk 2048: the memory rule's setting beside the 76.6 GiB rank files).
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "comm/ring.h"
#include "comm/ring_config.h"
#include "common/check.h"
#include "model/model.h"
#include "server/cmd.h"
#include "server/disk.h"
#include "server/keepfile.h"
#include "server/protocol.h"
#include "server/serve.h"
#include "weights/glw.h"
#include "weights/loader.h"

using namespace srv;

namespace {

constexpr double kGiB = 1073741824.0;

std::string expand(const std::string& p) {
  const char* h = std::getenv("HOME");
  return p.rfind("~/", 0) == 0 && h ? std::string(h) + p.substr(1) : p;
}

size_t meminfo(const char* key) {  // /proc/meminfo's key, bytes
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kb = 0;
  const size_t kl = std::strlen(key);
  while (std::fgets(line, sizeof line, f))
    if (std::strncmp(line, key, kl) == 0 && line[kl] == ':') {
      kb = std::strtoull(line + kl + 1, nullptr, 10);
      break;
    }
  std::fclose(f);
  return kb << 10;
}

// The rank file's pages in the page cache dropped before the load (posix_fadvise DONTNEED, no root needed): a copy
// of the file leaves up to its whole size cached (24 and 16.5 GiB on ranks 1 and 2 on 2026-10-14), memory CUDA's
// free count does not see; the loader reads with O_DIRECT, so the cached pages serve nothing.
void drop_cache(const std::string& path, int rk) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;  // the loader reports a missing file
  const size_t before = meminfo("MemFree");
  const int e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
  std::printf("rank %d: the rank file's page cache dropped%s: MemFree %.2f -> %.2f GiB\n", rk, e ? " (failed)" : "",
              before / 1073741824.0, meminfo("MemFree") / 1073741824.0);
}

struct Args {
  int rank = -1;
  std::string fleet, mode, weights = "fleet", socket = fe::kDefaultSocket, name = "altar-1";
  bool synthetic = false, fill_experts = false;
  int pools = 16;
  uint64_t seed = 1;
  int max_ctx = 262144, page = 1024, slots = 4, slice = 64, chunk = 8192, splits = 16, grow = 1024;
  double kv_gib = 1e9, margin_gib = 8, worker_margin_gib = 8, min_free_gib = 8;
  double disk_gib = 32, disk_margin_gib = 32, disk_rank0_gib = 0, keep_gib = 0, keep_days = 7;
  std::string disk_dir, disk_tag;
  int step_timeout = 120;
  int index_scratch_mib = 512;  // the prefill indexer's score scratch (Options::index_scratch)
  bool pool_free = false;  // --pool-basis free: the pool from min(free device memory, MemAvailable), not MemAvailable
  bool spec = true;
  int spec_drafts = 3, spec_force = 0;
  bool spec_oracle = false;
  float spec_c1 = 52.4f, spec_drow = 13.0f;
};

// What a kept session's caches depend on: this binary's bytes, the weights (path, size, time), the synthetic
// experts' options, and the geometry. A start whose id differs from the stop's starts clean.
uint64_t run_id(const Args& a, const std::string& weights, size_t pos_bytes) {
  uint64_t h = keepfile::fnv("altar1 kept sessions", 20);
  const int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  if (fd < 0) die("run id: /proc/self/exe: %s", std::strerror(errno));
  std::vector<uint8_t> buf(1 << 20);
  for (ssize_t r; (r = read(fd, buf.data(), buf.size())) > 0;) h = keepfile::fnv(buf.data(), (size_t)r, h);
  close(fd);
  struct stat s{};
  const int64_t v[3] = {stat(weights.c_str(), &s) == 0 ? (int64_t)s.st_size : -1, (int64_t)s.st_mtim.tv_sec,
                        (int64_t)s.st_mtim.tv_nsec};
  h = keepfile::fnv(weights.data(), weights.size(), h);
  h = keepfile::fnv(v, sizeof v, h);
  const int64_t o[] = {a.synthetic, a.fill_experts, a.pools, (int64_t)a.seed, a.page, (int64_t)pos_bytes};
  return keepfile::fnv(o, sizeof o, h);
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", s.c_str());
      return argv[++i];
    };
    if (s == "--rank") a.rank = std::stoi(val());
    else if (s == "--fleet") a.fleet = val();
    else if (s == "--mode") a.mode = val();
    else if (s == "--weights") a.weights = val();
    else if (s == "--synthetic") a.synthetic = true;
    else if (s == "--fill-experts") a.fill_experts = true;
    else if (s == "--pools") a.pools = std::stoi(val());
    else if (s == "--seed") a.seed = std::stoull(val());
    else if (s == "--socket") a.socket = val();
    else if (s == "--name") a.name = val();
    else if (s == "--max-ctx") a.max_ctx = std::stoi(val());
    else if (s == "--page") a.page = std::stoi(val());
    else if (s == "--slots") a.slots = std::stoi(val());
    else if (s == "--slice") a.slice = std::stoi(val());
    else if (s == "--chunk") a.chunk = std::stoi(val());
    else if (s == "--splits") a.splits = std::stoi(val());
    else if (s == "--grow") a.grow = std::stoi(val());
    else if (s == "--kv-gib") a.kv_gib = std::atof(val().c_str());
    else if (s == "--margin-gib") a.margin_gib = std::atof(val().c_str());
    else if (s == "--worker-margin-gib") a.worker_margin_gib = std::atof(val().c_str());
    else if (s == "--min-free-gib") a.min_free_gib = std::atof(val().c_str());
    else if (s == "--pool-basis") { const std::string v = val(); if (v != "free" && v != "avail") die("--pool-basis free|avail"); a.pool_free = v == "free"; }
    else if (s == "--disk-gib") a.disk_gib = std::atof(val().c_str());
    else if (s == "--disk-margin-gib") a.disk_margin_gib = std::atof(val().c_str());
    else if (s == "--disk-rank0-gib") a.disk_rank0_gib = std::atof(val().c_str());
    else if (s == "--disk-dir") a.disk_dir = val();
    else if (s == "--disk-tag") a.disk_tag = val();
    else if (s == "--keep-gib") a.keep_gib = std::atof(val().c_str());
    else if (s == "--keep-days") a.keep_days = std::atof(val().c_str());
    else if (s == "--step-timeout") a.step_timeout = std::stoi(val());
    else if (s == "--index-scratch-mib") a.index_scratch_mib = std::stoi(val());
    else if (s == "--no-spec") a.spec = false;
    else if (s == "--spec-drafts") a.spec_drafts = std::stoi(val());
    else if (s == "--spec-force") a.spec_force = std::stoi(val());
    else if (s == "--spec-oracle") a.spec_oracle = true;
    else if (s == "--spec-cost") {
      const std::string v = val();
      if (std::sscanf(v.c_str(), "%f,%f", &a.spec_c1, &a.spec_drow) != 2 || !(a.spec_c1 > 0) || !(a.spec_drow > 0))
        die("--spec-cost C1,DROW (ms, both > 0)");
    }
    else die("unknown option %s (the header of engine/server/main.cpp lists them)", s.c_str());
  }
  if (a.mode != "serve") die("altar1_rank: --mode serve");
  if (a.page < 256 || a.page % 256) die("--page: a multiple of 256");
  if (a.chunk < 16 || a.chunk > model::kMaxChunk || a.chunk % 16) die("--chunk: 16 .. %d, a multiple of 16", model::kMaxChunk);
  if (a.disk_tag.size() > 32 || a.disk_tag.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
    die("--disk-tag: at most 32 of a-z, 0-9, _ and -");
  if (a.keep_gib < 0 || !(a.keep_days > 0)) die("--keep-gib must be >= 0 and --keep-days > 0");
  if (a.min_free_gib < 0 || a.disk_margin_gib < 0 || a.disk_rank0_gib < 0) die("--min-free-gib, --disk-margin-gib, --disk-rank0-gib: >= 0");
  if (a.spec_drafts < 1 || a.spec_drafts > model::kMaxDrafts) die("--spec-drafts: 1 .. %d", model::kMaxDrafts);
  if (!a.spec) a.spec_drafts = 0;
  if (a.spec && a.splits == 0) die("--splits 0 (split-K by row count) with drafts: verify steps need a fixed split-K");
  if (a.spec_force < 0 || a.spec_force > a.spec_drafts) die("--spec-force: 0 .. --spec-drafts");
  g_spec.kmax = a.spec_drafts, g_spec.cost.c1 = a.spec_c1, g_spec.cost.drow = a.spec_drow, g_spec.force = a.spec_force;
  g_spec.oracle = a.spec_oracle && a.spec;
  const double t0 = now_s();
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  comm::Options co;
  co.fleet = a.fleet;
  co.rank = a.rank;
  co.cap = 4 << 20;
  co.bootstrap_timeout_s = 300;
  comm::Ring ring(co);
  const int rk = ring.rank();
  a.rank = rk;
  const comm::Fleet& fl = comm::fleet(a.fleet);
  if (a.disk_dir.empty()) a.disk_dir = comm::fleet_path(fl.rank[rk].spill, rk);
  a.disk_dir = expand(a.disk_dir);

  // 1. the model: a pool of one page for now (resized below), the weights, the prefill buffers
  model::Options mo;
  mo.max_ctx = a.max_ctx, mo.page = a.page, mo.slots = 1, mo.prefill_rows = a.chunk, mo.attn_splits = a.splits;
  mo.drafts = a.spec_drafts;
  mo.index_scratch = (size_t)a.index_scratch_mib << 20;
  model::Model m(rk, mo);
  std::unique_ptr<glw::WeightFile> wf;
  model::SynthOptions so;
  so.seed = a.seed, so.expert_pools = a.pools;
  std::string wpath = "synthetic";
  double file_gib = 0;  // the rank file's weights with its reserved experts at their reserved size
  if (a.synthetic) {
    m.bind_synthetic(so, ring);
  } else {
    char path[1024];
    if (a.weights == "fleet") std::snprintf(path, sizeof path, "%s", comm::fleet_path(fl.rank[rk].weights, rk).c_str());
    else std::snprintf(path, sizeof path, expand(a.weights).c_str(), rk);
    wpath = path;
    drop_cache(wpath, rk);  // the file's page cache (a copy leaves it; the loader reads with O_DIRECT)
    wf = std::make_unique<glw::WeightFile>(wpath, rk, model::kRanks);
    for (const glw::DeviceTensor& t : wf->tensors()) file_gib += (t.reserved() ? t.reserve_bytes : t.bytes) / kGiB;
    wf->load();
    m.bind(*wf, ring, a.fill_experts ? &so : nullptr);
  }
  m.prefill_init();
  CUDA_CHECK(cudaDeviceSynchronize());

  // the memory guard (the fleet's memory rule): every rank's MemAvailable after the weights and the prefill buffers;
  // all stop when one has less than --min-free-gib and one page left (the pool could not be had without going under)
  const size_t pos_bytes0 = m.cache_pos_bytes();
  {
    size_t df = 0, dt = 0;
    CUDA_CHECK(cudaMemGetInfo(&df, &dt));
    const double mine[2] = {df / kGiB, meminfo("MemAvailable") / kGiB};
    double all[4][2];
    ring.allgather_host(mine, all, sizeof mine);
    const double need = a.min_free_gib + (double)pos_bytes0 * a.page / kGiB;
    bool ok = true;
    for (const auto& r : all) ok = ok && r[1] >= need;
    if (rk == 0)
      std::printf("memory after the weights and the prefill buffers (chunk %d, index scratch %d MiB), free device / "
                  "MemAvailable, ranks 0-3: %.2f / %.2f, %.2f / %.2f, %.2f / %.2f, %.2f / %.2f GiB; the guard: %.2f GiB "
                  "(--min-free-gib %.1f and a page)\n",
                  a.chunk, a.index_scratch_mib, all[0][0], all[0][1], all[1][0], all[1][1], all[2][0], all[2][1],
                  all[3][0], all[3][1], need, a.min_free_gib);
    if (!ok)
      die("rank %d: the memory guard: a rank has less than %.2f GiB available after the weights (MemAvailable, ranks "
          "0-3: %.2f %.2f %.2f %.2f GiB)", rk, need, all[0][1], all[1][1], all[2][1], all[3][1]);
  }

  // 2. the disk tier first (its staging is pinned host memory), with the kept sessions of a planned stop
  const size_t pos_bytes = m.cache_pos_bytes(), page_bytes = pos_bytes * (size_t)a.page;
  const std::string base = a.disk_dir + "/rank" + std::to_string(rk) + (a.disk_tag.empty() ? "" : "." + a.disk_tag);
  const uint64_t id = run_id(a, wpath, pos_bytes);
  std::unique_ptr<Disk> disk;
  uint64_t kept_gen = 0;
  if (a.disk_gib > 0) {
    std::error_code ec;
    std::filesystem::create_directories(a.disk_dir, ec);
    chmod(a.disk_dir.c_str(), 0700);
    struct statvfs v{};
    uint64_t fr = statvfs(a.disk_dir.c_str(), &v) == 0 ? (uint64_t)v.f_bavail * v.f_frsize : 0;
    struct stat st{};
    if (stat((base + ".spill").c_str(), &st) == 0) fr += (uint64_t)st.st_size;  // its own file counts as free
    const uint64_t margin = (uint64_t)(a.disk_margin_gib * kGiB);
    std::string why;
    int kpages = 0;
    const uint64_t gen = a.keep_gib > 0 ? Disk::kept_check(base, id, page_bytes, a.keep_days * 86400, &kpages, &why) : 0;
    if (!why.empty()) std::printf("rank %d: kept sessions: none reused (%s)\n", rk, why.c_str());
    const double cap_gib = rk == 0 ? std::min(a.disk_gib, a.disk_rank0_gib) : a.disk_gib;  // rank 0's data disk
    const uint64_t mine[3] = {std::min<uint64_t>((uint64_t)(cap_gib * kGiB), fr > margin ? fr - margin : 0) / page_bytes,
                              gen, (uint64_t)kpages};
    uint64_t all[4][3];
    ring.allgather_host(mine, all, sizeof mine);
    uint64_t pages = mine[0];
    bool reuse = gen != 0;
    for (const auto& x : all) pages = std::min(pages, x[0]), reuse = reuse && x[1] == gen && x[2] == (uint64_t)kpages;
    reuse = reuse && (uint64_t)kpages <= std::max<uint64_t>(pages, kpages) && kpages > 0;
    if (reuse) pages = kpages;
    if (pages > 0) {
      disk = std::make_unique<Disk>(base, (int)pages, page_bytes, reuse, id);
      kept_gen = reuse ? gen : 0;
    } else {
      unlink((base + ".index").c_str());
      unlink((base + ".spill").c_str());
    }
    if (rk == 0)
      std::printf("disk tier: %llu pages, %.1f GiB a rank in %s (offers, ranks 0-3: %llu %llu %llu %llu pages; rank 0 at "
                  "most --disk-rank0-gib %.1f, every rank its free space less %.1f GiB); kept sessions: %s\n",
                  (unsigned long long)pages, pages * (double)page_bytes / kGiB, base.c_str(), (unsigned long long)all[0][0],
                  (unsigned long long)all[1][0], (unsigned long long)all[2][0], (unsigned long long)all[3][0],
                  a.disk_rank0_gib, a.disk_margin_gib,
                  reuse ? "the files of the last planned stop are reused" : "none, a clean start");
  }

  // 3. the cache pool: every rank's offer, the smallest taken
  size_t dfree = 0, dtotal = 0;
  CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
  const size_t avail = meminfo("MemAvailable"), total = meminfo("MemTotal");
  const size_t margin = (size_t)(std::max(rk == 0 ? a.margin_gib : a.worker_margin_gib, a.min_free_gib) * kGiB);
  const size_t basis = a.pool_free ? std::min(dfree, avail) : avail;  // MemAvailable: CUDA reclaims page cache (§ 3)
  const size_t offer = basis > margin ? basis - margin : 0;
  const double wgib = m.weight_bytes() / kGiB;
  const double mine[6] = {(double)offer, dfree / kGiB, avail / kGiB, wgib, file_gib, (double)m.file_expert_layers()};
  double all[4][6];
  ring.allgather_host(mine, all, sizeof mine);
  size_t pool = (size_t)std::min(a.kv_gib * kGiB, 1e18);
  for (const auto& r : all) pool = std::min(pool, (size_t)r[0]);
  const int pages = (int)std::min<size_t>(pool / page_bytes, model::kMaxPages);
  if (pages < 1) die("rank %d: no memory for the cache pool (offers: %.2f %.2f %.2f %.2f GiB)", rk, all[0][0] / kGiB,
                     all[1][0] / kGiB, all[2][0] / kGiB, all[3][0] / kGiB);
  m.resize_slots(pages);
  CUDA_CHECK(cudaDeviceSynchronize());
  double after[4][2];  // free device memory and MemAvailable once the pool is allocated: what the memory rule keeps
  {
    size_t df = 0, dt = 0;
    CUDA_CHECK(cudaMemGetInfo(&df, &dt));
    const double mine2[2] = {df / kGiB, meminfo("MemAvailable") / kGiB};
    ring.allgather_host(mine2, after, sizeof mine2);
  }
  if (rk == 0) {
    const int need = (a.max_ctx + a.page - 1) / a.page;  // a session of --max-ctx positions
    std::printf("cache pool: a session of %d positions (%d pages, %.2f GiB a rank) %s; free device / MemAvailable after "
                "the pool, ranks 0-3: %.2f / %.2f, %.2f / %.2f, %.2f / %.2f, %.2f / %.2f GiB\n",
                a.max_ctx, need, need * (double)page_bytes / kGiB, pages >= need ? "fits" : "does NOT fit (the context is capped by the pool)",
                after[0][0], after[0][1], after[1][0], after[1][1], after[2][0], after[2][1], after[3][0], after[3][1]);
    std::printf("cache pool: %d pages of %d positions = %d positions, %.2f GiB a rank (%zu B a position); free device / "
                "available host memory before it, ranks 0-3:",
                pages, a.page, pages * a.page, pages * (double)page_bytes / kGiB, pos_bytes);
    for (const auto& r : all) std::printf(" %.2f / %.2f", r[1], r[2]);
    std::printf(" GiB; the offer: %s less margins of %.1f (rank 0) and %.1f GiB\n",
                a.pool_free ? "min(free device, MemAvailable)" : "MemAvailable", a.margin_gib, a.worker_margin_gib);
    std::printf("weights: %s, %.2f GiB a rank bound (%d MoE layers' experts from the file); the file at its reserved "
                "size: %.2f GiB; MemTotal %.2f GiB\n",
                wpath.c_str(), wgib, m.file_expert_layers(), file_gib, total / kGiB);
    for (const auto& r : all)
      if (r[5] != all[0][5])  // a rank file filled and another not: the layer's experts differ by rank
        std::printf("weights: WARNING: the ranks' files differ in MoE layers with their experts from the file (ranks 0-3: "
                    "%.0f %.0f %.0f %.0f): the text is wrong\n", all[0][5], all[1][5], all[2][5], all[3][5]);
    std::printf("drafts: %s\n", a.spec ? ("up to " + std::to_string(a.spec_drafts) + " a session a step" +
                                           (a.spec_force ? ", forced to " + std::to_string(a.spec_force) : std::string()) +
                                           (a.spec_oracle ? ", oracle drafts for repeated requests" : "") +
                                           ", gate cost " +
                                           std::to_string(a.spec_c1) + " + " + std::to_string(a.spec_drow) + " ms a row").c_str()
                                        : "off (--no-spec)");
  }
  ring.barrier();
  pin_self(15);
  std::printf("rank %d: ready after %.1f s\n", rk, now_s() - t0);

  Watchdog wd(a.step_timeout, ring);
  if (rk != 0) {
    follow(m, ring, wd, disk.get(), id);
    std::printf("rank %d: done\n", rk);
    return 0;
  }
  ServeOptions s;
  s.socket = a.socket, s.name = a.name, s.max_ctx = a.max_ctx, s.chunk = a.chunk, s.slots = a.slots, s.slice = a.slice;
  s.grow = a.grow;
  s.keep_index = a.disk_dir + "/server" + (a.disk_tag.empty() ? "" : "." + a.disk_tag) + ".index";
  s.keep_bytes = disk ? std::min<size_t>((size_t)disk->pages() * page_bytes, (size_t)(a.keep_gib * kGiB)) : 0;
  s.keep_age_s = a.keep_days * 86400;
  s.kept_gen = kept_gen;
  s.run_id = id;
  const int rc = serve(m, ring, wd, disk.get(), s);
  std::printf("rank 0: done\n");
  return rc;
}
