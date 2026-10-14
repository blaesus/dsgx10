// compile-only probe: does ptxas accept kind::mxf8f6f4 with ue4m3 (E4M3) block scales on sm_121a? (build.sh records it)
#include <cstdint>
__global__ void k(float* o, uint32_t a0, uint32_t b0, uint32_t s) {
  float d[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3}, %6, {0, 0}, %6, {0, 0};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(b0), "r"(s));
  o[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
}
