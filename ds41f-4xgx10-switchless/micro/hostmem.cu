// GPU read speed and RDMA registrability of the host-memory kinds a collective buffer can use:
// cudaMalloc (reference), cudaHostAlloc, and cuMemCreate with a HOST_NUMA location.
// build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o hostmem hostmem.cu -lcuda -libverbs
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); printf("%s:%d %s\n", __FILE__, __LINE__, s); exit(1); } } while (0)

__global__ void rd(const uint4* p, size_t n, uint4* sink) {
  uint4 a = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
  }
  if (a.x == 0x12345678) *sink = a;
}
// dependent small reads: latency of one 16 B load
__global__ void chase(const uint4* p, int iters, uint4* sink) {
  uint32_t idx = 0;
  for (int i = 0; i < iters; i++) { uint4 v = __ldcv(p + idx); idx = (v.x + i * 4099) & 0xffff; }
  sink->x = idx;
}

static float time_read(const void* p, size_t bytes, uint4* sink) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  rd<<<192, 512>>>((const uint4*)p, bytes / 16, sink);
  CK(cudaDeviceSynchronize());
  float best = 1e9;
  for (int r = 0; r < 5; r++) {
    cudaEventRecord(a);
    rd<<<192, 512>>>((const uint4*)p, bytes / 16, sink);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
  }
  return bytes / (best * 1e6);
}
static float time_chase(const void* p, uint4* sink) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int iters = 20000;
  chase<<<1, 1>>>((const uint4*)p, 100, sink); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); chase<<<1, 1>>>((const uint4*)p, iters, sink); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000 / iters;
}
static void try_reg(const char* what, void* p, size_t bytes) {
  int n; ibv_device** l = ibv_get_device_list(&n);
  ibv_context* ctx = 0;
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(l[i]), "rocep1s0f0")) ctx = ibv_open_device(l[i]);
  if (!ctx) { printf("  %s: no rocep1s0f0\n", what); return; }
  ibv_pd* pd = ibv_alloc_pd(ctx);
  ibv_mr* mr = ibv_reg_mr(pd, p, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  printf("  %s: ibv_reg_mr %s\n", what, mr ? "ok" : strerror(errno));
  if (mr) ibv_dereg_mr(mr);
  ibv_dealloc_pd(pd); ibv_close_device(ctx); ibv_free_device_list(l);
}

int main() {
  size_t bytes = 1ull << 30;
  uint4* sink; CK(cudaMalloc(&sink, 64));
  CU(cuInit(0));
  void* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 1, bytes));
  printf("cudaMalloc:     %.1f GB/s, dependent 16 B load %.2f us\n", time_read(d, bytes, sink), time_chase(d, sink));
  void* h; CK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped)); memset(h, 1, bytes);
  printf("cudaHostAlloc:  %.1f GB/s, dependent 16 B load %.2f us\n", time_read(h, bytes, sink), time_chase(h, sink));
  try_reg("cudaHostAlloc", h, bytes);

  // cuMemCreate on the host NUMA node, mapped for both the device and the host
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = 0;
  size_t gran; CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t sz = (bytes + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle hdl; CU(cuMemCreate(&hdl, sz, &prop, 0));
  CUdeviceptr va; CU(cuMemAddressReserve(&va, sz, 0, 0, 0)); CU(cuMemMap(va, sz, 0, hdl, 0));
  CUmemAccessDesc acc[2]{};
  acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc[0].location.id = 0; acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  acc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; acc[1].location.id = 0; acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUresult r = cuMemSetAccess(va, sz, acc, 2);
  if (r != CUDA_SUCCESS) { printf("  cuMemSetAccess(device+host) failed (%d); device only\n", (int)r); CU(cuMemSetAccess(va, sz, acc, 1)); }
  CK(cudaMemset((void*)va, 1, bytes));
  printf("cuMemCreate HOST_NUMA (granularity %zu): %.1f GB/s, dependent 16 B load %.2f us\n", gran, time_read((void*)va, bytes, sink), time_chase((void*)va, sink));
  if (r == CUDA_SUCCESS) {
    volatile char* c = (volatile char*)va; c[0] = 2; printf("  host access: ok (%d)\n", c[0]);
    try_reg("cuMemCreate HOST_NUMA", (void*)va, bytes);
  }
  return 0;
}
