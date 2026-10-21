#include "server/vision_prefix.h"
#include <cassert>
#include <iostream>
int main() {
  using fe::ImageSpan;
  for (uint32_t f=0; f<8; ++f) assert(fe::valid_submit_flags(f));
  for (uint32_t f : {8u,16u,32u,0x80000000u}) assert(!fe::valid_submit_flags(f));
  ImageSpan a{100,32,{1}}, b{500,32,{2}}, c{900,32,{3}};
  using V=std::vector<ImageSpan>;
  assert(srv::vision_prefix_limit(V{a,b},V{a,b,c})==900);
  auto replacement=b; replacement.digest[0]=7;
  assert(srv::vision_prefix_limit(V{a,b,c},V{a,replacement,c})==500);
  assert(srv::vision_prefix_limit(V{a,b,c},V{a,c})==500);
  assert(srv::vision_prefix_limit(V{a,b,c},V{a,b})==900);
  auto moved=b;moved.start=480;
  assert(srv::vision_prefix_limit(V{a,b},V{a,moved})==480);
  auto resized=b;resized.rows=64;
  assert(srv::vision_prefix_limit(V{a,b},V{a,resized})==500);
  assert(srv::vision_prefix_limit(V{a,b},V{a,b})==INT_MAX);
  assert(srv::vision_prefix_limit(V{},V{a})==0);
  assert(srv::vision_prefix_limit(V{a},V{})==0);
  // The first changed image never belongs to the prefix, including forged
  // all-equal placeholder token streams and arbitrary claimed prefix hints.
  for(int n=1;n<=64;++n) {
    V old;
    for(int i=0;i<n;++i) old.push_back(ImageSpan{uint32_t(100+i*64),32,{uint8_t(i)}});
    for(int j=0;j<n;++j) {
      auto next=old;next[j].digest[31]^=0xff;
      assert(srv::vision_prefix_limit(old,next)==int(old[j].start));
    }
  }
  for(int q=0;q<200;++q) {
    ImageSpan image{100,32,{1}};
    const bool selected=srv::vision_needs_replay(image,q);
    for(int pos=q;pos<q+8;++pos) for(int look=0;look<4;++look) {
      const int at=std::max(0,pos-look);
      const bool full_mask=at<100 || at>=132;
      const bool pruned_mask=!selected || full_mask;
      assert(full_mask==pruned_mask);
    }
  }
  std::cout<<"vision_prefix_test passed\n";
}
