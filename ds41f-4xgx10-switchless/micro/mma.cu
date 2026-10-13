// Verify mma.sync kind::mxf8f6f4 block_scale (e4m3 x e2m1 / e4m3 x e4m3, ue8m0 per 32) on sm_121a
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float e2m1_val(int c){ static const float t[16]={0,0.5,1,1.5,2,3,4,6,-0.f,-0.5,-1,-1.5,-2,-3,-4,-6}; return t[c&15]; }
static float e4m3_val(uint8_t b){ __nv_fp8_e4m3 v; *(uint8_t*)&v=b; return float(v); }

__device__ __forceinline__ void mma_e4m3_e2m1(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb){
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),"r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
}
__device__ __forceinline__ void mma_e2m1_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb){
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),"r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
}
// A: 16x32 bytes row-major; B: 8x32 bytes (n-major, i.e. B^T row-major); sA[16], sB[8]; out D 16x8
template<bool AFP4>
__global__ void test(const uint8_t* A, const uint8_t* B, const uint8_t* sA, const uint8_t* sB, float* D){
  int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  uint32_t a[4], b[2];
  auto ld4 = [&](const uint8_t* p){ return *(const uint32_t*)p; };
  a[0] = ld4(A + g*32 + t*4); a[1] = ld4(A + (g+8)*32 + t*4); a[2] = ld4(A + g*32 + 16 + t*4); a[3] = ld4(A + (g+8)*32 + 16 + t*4);
  b[0] = ld4(B + g*32 + t*4); b[1] = ld4(B + g*32 + 16 + t*4);
  uint32_t sfa = sA[g + 8*(lane & 1)], sfb = sB[g];
  float d[4] = {0,0,0,0};
  if (AFP4) mma_e2m1_e4m3(d, a, b, sfa, sfb); else mma_e4m3_e2m1(d, a, b, sfa, sfb);
  D[g*8 + t*2] = d[0]; D[g*8 + t*2+1] = d[1]; D[(g+8)*8 + t*2] = d[2]; D[(g+8)*8 + t*2+1] = d[3];
}
// throughput: many independent MMAs
__global__ void thru(float* out, int iters){
  uint32_t a[4]={0x38383838u,0x38383838u,0x38383838u,0x38383838u}, b[2]={0x08080808u,0x08080808u};
  float d0[4]={0,0,0,0}, d1[4]={0,0,0,0}, d2[4]={0,0,0,0}, d3[4]={0,0,0,0};
  for (int i=0;i<iters;i++){ mma_e4m3_e2m1(d0,a,b,127,127); mma_e4m3_e2m1(d1,a,b,127,127); mma_e4m3_e2m1(d2,a,b,127,127); mma_e4m3_e2m1(d3,a,b,127,127); }
  if (d0[0]+d1[1]+d2[2]+d3[3] == 1.2345f) out[0]=1;
}
int main(){
  std::mt19937 rng(1);
  for (int mode=0; mode<2; mode++) {
    bool afp4 = mode==1;
    std::vector<uint8_t> A(16*32), B(8*32), sA(16), sB(8); std::vector<float> D(128), R(128,0);
    auto rnd_fp8 = [&](){ uint8_t v; do { v = rng() & 0xff; } while ((v&0x7f)==0x7f); return v; };
    auto rnd_fp4 = [&](){ return (uint8_t)((rng() & 15) << 2); };
    for (auto& x: A) x = afp4 ? rnd_fp4() : rnd_fp8();
    for (auto& x: B) x = afp4 ? rnd_fp8() : rnd_fp4();
    for (auto& x: sA) x = 120 + rng()%14; for (auto& x: sB) x = 120 + rng()%14;
    auto val = [&](uint8_t x, bool fp4){ return fp4 ? e2m1_val(x>>2) : e4m3_val(x); };
    for (int m=0;m<16;m++) for (int n=0;n<8;n++){ double s=0; for(int k=0;k<32;k++) s += (double)val(A[m*32+k], afp4)*val(B[n*32+k], !afp4); R[m*8+n] = s*ldexp(1.0, sA[m]-127)*ldexp(1.0, sB[n]-127); }
    uint8_t *dA,*dB,*dsA,*dsB; float* dD;
    CK(cudaMalloc(&dA,512)); CK(cudaMalloc(&dB,256)); CK(cudaMalloc(&dsA,16)); CK(cudaMalloc(&dsB,8)); CK(cudaMalloc(&dD,512));
    cudaMemcpy(dA,A.data(),512,cudaMemcpyHostToDevice); cudaMemcpy(dB,B.data(),256,cudaMemcpyHostToDevice);
    cudaMemcpy(dsA,sA.data(),16,cudaMemcpyHostToDevice); cudaMemcpy(dsB,sB.data(),8,cudaMemcpyHostToDevice);
    if (afp4) test<true><<<1,32>>>(dA,dB,dsA,dsB,dD); else test<false><<<1,32>>>(dA,dB,dsA,dsB,dD);
    CK(cudaDeviceSynchronize()); cudaMemcpy(D.data(),dD,512,cudaMemcpyDeviceToHost);
    double maxrel=0; for(int i=0;i<128;i++){ double e=fabs(D[i]-R[i])/(fabs(R[i])+1e-6); if(e>maxrel) maxrel=e; }
    printf("%s: max rel err %.3g  (D[0]=%g R[0]=%g D[77]=%g R[77]=%g)\n", afp4?"e2m1 x e4m3":"e4m3 x e2m1", maxrel, D[0],R[0],D[77],R[77]);
  }
  float* o; cudaMalloc(&o,4); int iters=20000, blocks=48*8;
  thru<<<blocks,128>>>(o,10); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
  thru<<<blocks,128>>>(o,iters); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  double flops = 2.0*16*8*32*4.0*iters*blocks*4; printf("mxf8f6f4 e4m3xe2m1 throughput: %.1f TFLOPS\n", flops/(ms*1e9));
}
