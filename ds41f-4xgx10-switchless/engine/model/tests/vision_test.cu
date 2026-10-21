#include "model/ops.h"
#include "kernels/dsk.h"
#include "common/check.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
using namespace model;
template<class T> T* device(const std::vector<T>& h) {
  T* d=nullptr; CUDA_CHECK(cudaMalloc(&d,h.size()*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return d;
}
template<class T> std::vector<T> host(T* d,size_t n) {
  std::vector<T> h(n); CUDA_CHECK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost)); return h;
}
int main() {
  constexpr int R=6;
  auto pos=device<int>({9,10,11,12,13,20});
  auto spans=device<ops::VisionSpan>({{10,3,0},{20,1,3}});
  std::vector<ops::bf16> f(4*kDim), residual(R*kHcDim,__float2bfloat16(3.f));
  for(int r=0;r<4;++r) for(int j=0;j<kDim;++j) f[r*kDim+j]=__float2bfloat16(float(r+1)+float(j%7)/16);
  auto features=device(f),x=device(residual);auto mask=device<int>(std::vector<int>(R,-2));
  ops::vision_rows(pos,R,spans,2,features,x,mask,nullptr);
  const std::vector<int> expected={0,1,1,1,0,1};assert(host(mask,R)==expected);
  auto got=host(x,R*kHcDim);
  const int source[R]={-1,0,1,2,-1,3};
  for(int r=0;r<R;++r) for(int c=0;c<kHc;++c) for(int j=0;j<kDim;++j)
    assert(__bfloat162float(got[r*kHcDim+c*kDim+j])==__bfloat162float(source[r]<0?residual[r*kHcDim+c*kDim+j]:f[source[r]*kDim+j]));
  // CED replay can select different rows from the previous encoder chunk. Refresh
  // its mask without replacing the already-computed layer-20 residual.
  std::vector<int> replay={20,21,8,9,10,11};
  CUDA_CHECK(cudaMemcpy(pos,replay.data(),R*4,cudaMemcpyHostToDevice));
  ops::vision_rows(pos,R,spans,2,features,nullptr,mask,nullptr);
  assert(host(mask,R)==std::vector<int>({1,0,0,0,1,1}));
  assert(host(x,R*kHcDim).size()==got.size());
  auto replay_x=host(x,R*kHcDim);
  for(size_t i=0;i<got.size();++i)assert(__bfloat162float(replay_x[i])==__bfloat162float(got[i]));
  // Vision route selection changes the selection bias, not the probability
  // weights. Text and image rows in the same launch must select different sets.
  std::vector<float> b(kExperts),bv(kExperts);
  for(int e=0;e<6;++e)b[e]=100.f,bv[e+7]=100.f;
  auto bias=device(b),bias_vl=device(bv),logits=device<float>(std::vector<float>(R*kExperts));
  auto ids=device<int>(std::vector<int>(R*kTopK));auto weights=device<float>(std::vector<float>(R*kTopK));
  dsk::router_topk(logits,bias,R,kExperts,ids,weights,kRouteScale,nullptr,bias_vl,mask);
  auto routes=host(ids,R*kTopK);auto w=host(weights,R*kTopK);auto live=host(mask,R);
  for(int r=0;r<R;++r) {
    std::sort(routes.begin()+r*kTopK,routes.begin()+(r+1)*kTopK);
    for(int e=0;e<kTopK;++e) {assert(routes[r*kTopK+e]==e+(live[r]?7:0));assert(fabsf(w[r*kTopK+e]-0.25f)<1e-6f);}
  }
  // For image rows the Engram contribution is exactly zero, including delimiters.
  // A null gather is intentional: masked rows must not read any Engram values.
  CUDA_CHECK(cudaMemset(mask,1,R*4));
  ops::engram_combine(x,nullptr,R,x,nullptr,mask);
  auto masked=host(x,R*kHcDim);
  for(size_t i=0;i<got.size();++i)assert(__bfloat162float(masked[i])==__bfloat162float(got[i]));
  CUDA_CHECK(cudaDeviceSynchronize());
  // More than eight historical image spans, including the far end of a chunk.
  std::vector<ops::VisionSpan> history;
  std::vector<int> history_pos;
  std::vector<ops::bf16> history_features(64*kDim), history_x(64*kHcDim);
  for(int i=0;i<64;++i) {
    history.push_back({8000+i*3,1,i});history_pos.push_back(8000+i*3);
    for(int j=0;j<kDim;++j)history_features[i*kDim+j]=__float2bfloat16(float(i));
  }
  auto hs=device(history);auto hp=device(history_pos);auto hf=device(history_features);auto hx=device(history_x);
  auto hm=device<int>(std::vector<int>(64));
  ops::vision_rows(hp,64,hs,64,hf,hx,hm,nullptr);
  assert(host(hm,64)==std::vector<int>(64,1));
  auto hh=host(hx,64*kHcDim);
  for(int i=0;i<64;++i)for(int j=0;j<kHcDim;++j)assert(__bfloat162float(hh[i*kHcDim+j])==float(i));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::puts("VISION_KERNEL_TEST_PASS: embeddings, chunk boundaries, replay mask, per-row routing, Engram zero contribution");
}
