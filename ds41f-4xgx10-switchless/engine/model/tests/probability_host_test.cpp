// CPU-only contract tests of the actual shared RNG/metadata/verdict helpers.
// Distribution and CUDA/TP4 tests are independent qualification steps.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include "common/spec_sampling.h"
#ifdef DSGB10_TEST_WITH_CUDA_HEADERS
#include "model/spec_batch.h"
#endif

int main() {
  using namespace spec_sampling;
#ifdef DSGB10_TEST_WITH_CUDA_HEADERS
  using namespace model::ops;
#endif
  Params p{1017, 1.f, 1.f, 0, 0};
  assert(eligible(p));
  for (float t : {0.f, -1.f, INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
    auto x = p; x.temperature = t; assert(!eligible(x));
  }
  auto x = p; x.top_p = .9f; assert(!eligible(x));
  x = p; x.top_k = 5; assert(!eligible(x));
  ProposalMeta m{1017, 1.f, 100, 1, {0,0,0}};
  assert(!matches(m,p,99) && matches(m,p,100) && matches(m,p,104) && !matches(m,p,105));
  x=p; ++x.seed; assert(!matches(m,x,100));
  x=p; x.temperature=.5f; assert(!matches(m,x,100));
  m.valid=0; assert(!matches(m,p,100)); m.valid=-1; assert(!matches(m,p,100)); m.valid=1;
  for (uint64_t i=0; i<(1u<<23); ++i) {
    const float u=unit(i<<41); assert(u>0.f && u<1.f && std::isfinite(u));
  }
  for (int i=0;i<10000;++i) {
    const float d=uniform(1017,i,kDraftStream,i);
    assert(d==uniform(1017,i,kDraftStream,i));
    assert(d>0 && d<1);
  }
  assert(uniform(1017,100,kDraftStream,123) != uniform(1017,100,kResidualStream,123));
  assert(uniform(1017,100,kDraftStream) != uniform(1017,100,kAcceptStream));
  assert(std::fabs(std::exp(log_positive_difference(std::log(.6f),std::log(.2f)))-.4f)<1e-6f);
  assert(log_positive_difference(std::log(.2f),std::log(.2f))==-INFINITY);
  assert(log_positive_difference(-INFINITY,std::log(.2f))==-INFINITY);
  assert(log_positive_difference(std::log(.2f),-INFINITY)==std::log(.2f));
  assert(std::isfinite(log_positive_difference(-1.f,std::nextafter(-1.f,-INFINITY))));
#ifdef DSGB10_TEST_WITH_CUDA_HEADERS
  assert(spec_row_accepted(5,5,0) && !spec_row_accepted(6,5,0));
  assert(spec_row_accepted(5,5,kSampProb) && !spec_row_accepted(5,5,kSampProb|kSampProbReject));
  // First rejection terminates a session even when a later parallel row accepts.
  StepIn in{}; SampleRow rows[model::kMaxRows]{}; int32_t fb[model::kMaxRows]{};
  in.rows=6; in.nsess=2;
  for(int i=0;i<6;++i) {in.tokens[i]=20+i; in.pos[i]=100+i; in.sess[i]=i<3?0:1; fb[i]=21+i;}
  rows[0].flags=kSampProb|kSampProbReject;
  rows[1].flags=kSampProb;
  rows[3].flags=kSampProb; rows[4].flags=kSampProb;
  SpecVerdict v[kMaxSess]; spec_batch_verdicts(in,fb,true,v,rows);
  assert(v[0].rows==1 && v[0].bonus==21 && v[1].rows==3 && v[1].bonus==26);
  spec_batch_verdicts(in,fb,true,v); // legacy equality unchanged
  assert(v[0].rows==3 && v[1].rows==3);
  std::puts("probability_host_test: PASS (8388608 midpoint values, eligibility, metadata, residual, first rejection, batch isolation)");
#else
  std::puts("probability_host_test: PASS (8388608 midpoint values, eligibility, metadata, residual); batch verdict skipped: CUDA headers absent");
#endif
}
