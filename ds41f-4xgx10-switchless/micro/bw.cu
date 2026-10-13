// Streaming-read bandwidth microbenchmark for GB10.
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
__global__ void rd(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = make_int4(0,0,0,0);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
    int4 v = __ldg(p + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (acc.x == 0x12345678 && acc.y == 1) out[0] = acc.z + acc.w;
}
__global__ void rd_unroll(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = make_int4(0,0,0,0);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i + 3*stride < n; i += 4*stride) {
    int4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2*stride), d = __ldcs(p + i + 3*stride);
    acc.x ^= a.x ^ b.x ^ c.x ^ d.x; acc.y ^= a.y ^ b.y ^ c.y ^ d.y; acc.z ^= a.z^b.z^c.z^d.z; acc.w ^= a.w^b.w^c.w^d.w;
  }
  for (; i < n; i += stride) { int4 v = p[i]; acc.x ^= v.x; }
  if (acc.x == 0x12345678 && acc.y == 1) out[0] = acc.z + acc.w;
}
int main() {
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("%s SMs=%d L2=%d MB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize>>20);
  size_t bytes = (size_t)4 << 30;
  int4* p; int* out; CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&out, 4));
  CK(cudaMemset(p, 1, bytes));
  size_t n = bytes / 16;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int blocksPerSM : {2, 4, 8, 16}) for (int thr : {256, 512, 1024}) {
    if (blocksPerSM * thr > 2048) continue;
    int grid = prop.multiProcessorCount * blocksPerSM;
    for (int k = 0; k < 2; k++) {
      rd_unroll<<<grid, thr>>>(p, n, out); 
      cudaEventRecord(a);
      for (int it = 0; it < 5; it++) (k ? rd_unroll : rd)<<<grid, thr>>>(p, n, out);
      cudaEventRecord(b); CK(cudaEventSynchronize(b));
      float ms; cudaEventElapsedTime(&ms, a, b);
      printf("%s grid=%d thr=%d : %.1f GB/s\n", k ? "unroll4" : "simple ", grid, thr, 5.0 * bytes / (ms * 1e6));
    }
  }
  // memcpy D2D
  int4* q; CK(cudaMalloc(&q, bytes));
  cudaEventRecord(a); for (int it=0; it<3; it++) cudaMemcpy(q, p, bytes, cudaMemcpyDeviceToDevice); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b); printf("memcpy D2D: %.1f GB/s (copy), %.1f GB/s traffic\n", 3.0*bytes/(ms*1e6), 6.0*bytes/(ms*1e6));
  return 0;
}
