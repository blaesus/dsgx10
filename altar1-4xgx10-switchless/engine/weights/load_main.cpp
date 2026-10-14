// glw_load: time loading a rank's weight file onto the GPU, optionally checking every tensor's CRC on the device
// copy (the DeepSeek engine's dsw_load).
//
//   glw_load FILE [--threads 4] [--chunk-mib 64] [--limit-gib G] [--only SUBSTR] [--verify-crc] [--rank R] [--host]
// --rank R: refuse the file unless it is rank R of 4's, as the engine does.
// --host: no device: the loader opens the file (header, rank, TOC CRC, every extent) and reads every tensor
// through WeightFile::read_host, which checks its CRC32C (what a load does before any device work).
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#include "weights/io.h"
#include "weights/loader.h"

using namespace glw;

int main(int argc, char** argv) {
  if (argc < 2)
    die("usage: glw_load FILE [--threads 4] [--chunk-mib 64] [--limit-gib G] [--only SUBSTR] [--verify-crc] [--rank R]");
  std::string file = argv[1], only;
  LoadOptions opt;
  int rank = -1;
  bool host = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--threads") opt.threads = std::stoi(next());
    else if (a == "--chunk-mib") opt.chunk = std::stoull(next()) << 20;
    else if (a == "--limit-gib") opt.limit = uint64_t(std::stod(next()) * 1073741824.0);
    else if (a == "--verify-crc") opt.verify_crc = true;
    else if (a == "--only") only = next();
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--host") host = true;
    else die("unknown argument %s", a.c_str());
  }
  if (!only.empty()) opt.select = [&](const DeviceTensor& t) { return t.name.find(only) != std::string::npos; };
  WeightFile wf(file, rank, 4);  // header and TOC first: a wrong rank fails before any device work
  if (host) {
    const double t0 = now_s();
    uint64_t bytes = 0;
    size_t n = 0, reserved = 0;
    for (auto& t : wf.tensors()) {
      if (t.reserved()) { reserved++; continue; }
      if (!only.empty() && t.name.find(only) == std::string::npos) continue;
      bytes += wf.read_host(t).size();  // dies on a CRC mismatch
      n++;
    }
    const Header& h = wf.header();
    std::printf("%s: rank %u/%u, %u tensors (%zu reserved); host read of %zu tensors, %.3f GiB: every CRC32C OK "
                "(%.1f s)\n", file.c_str(), h.rank, h.world, h.n_tensors, reserved, n, bytes / 1073741824.0, now_s() - t0);
    return reserved ? 1 : 0;
  }
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));  // context creation outside the timing
  const Header& h = wf.header();
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  std::printf("%s: version %u, rank %u/%u, %u tensors, %.2f GiB of data; device %.1f GiB free of %.1f\n", file.c_str(),
              h.version, h.rank, h.world, h.n_tensors, h.data_bytes / 1073741824.0, free_b / 1073741824.0,
              total_b / 1073741824.0);
  LoadStats s = wf.load(opt);
  size_t loaded = 0, reserved = 0;
  for (auto& t : wf.tensors()) loaded += t.data != nullptr, reserved += t.reserved();
  std::printf("loaded %.2f GiB (%zu/%zu tensors, %zu reserved) in %.2f s: %.2f GB/s (%d threads, %llu MiB reads)\n",
              s.bytes / 1073741824.0, loaded, wf.tensors().size(), reserved, s.seconds, s.gbps, opt.threads,
              (unsigned long long)(opt.chunk >> 20));
  if (opt.verify_crc) std::printf("device copies match their CRCs (%.1f s)\n", s.verify_seconds);
  return 0;
}
