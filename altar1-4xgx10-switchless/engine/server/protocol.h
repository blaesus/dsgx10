// Front-end <-> rank-0 protocol over a Unix stream socket: frame layouts (docs/protocol.md; the DeepSeek engine's
// version 1 byte for byte, version 2's meanings: HELLO's vocab_size 154,856, eos_id one of the three stop ids).
// Every frame is a FrameHeader followed by `size` payload bytes: a fixed struct, then arrays.
// Little-endian, no padding between arrays. Plain C++, no CUDA.
#pragma once
#include <cstddef>
#include <cstdint>

namespace fe {

constexpr uint32_t kVersion = 2;
constexpr const char* kDefaultSocket = "/tmp/dsgb10-altar1-engine.sock";

enum MsgType : uint16_t {
  // front-end -> rank 0
  kHello = 0x01,    // HelloFe
  kSubmit = 0x02,   // SubmitHeader, uint32 stop_ids[n_stop_ids], uint32 prompt[n_prompt]
  kCancel = 0x03,   // Cancel
  kRelease = 0x04,  // Release
  // rank 0 -> front-end
  kHelloEngine = 0x81,  // HelloEngine, char name[name_len]
  kTokens = 0x82,       // TokensHeader, uint32 ids[n], [float logprob[n]], [TopLogprob top[n * n_top]]
  kDone = 0x83,         // DoneHeader, char error[error_len]
};

struct FrameHeader {
  uint32_t size;  // payload bytes after this header
  uint16_t type;  // MsgType
  uint16_t reserved;
};

struct HelloFe {
  uint32_t version;
  uint32_t reserved;
};

enum SubmitFlags : uint32_t {
  kFlagSeed = 1,      // use SubmitHeader::seed
  kFlagKeep = 2,      // keep the session's state after DONE
  kFlagLogprobs = 4,  // send logprobs in TOKENS
};

struct SubmitHeader {
  uint64_t req_id;           // unique per connection
  uint64_t session_id;       // state to continue or create; never 0
  uint64_t base_session_id;  // 0, or a resident session a new session may fork from (up to the LCP)
  uint64_t seed;
  float temperature;         // 0 = greedy
  float top_p;               // 1 = off
  uint32_t top_k;            // 0 = off
  uint32_t max_new_tokens;   // >= 1, already clamped to the context by the front-end
  uint32_t flags;            // SubmitFlags
  uint32_t n_top_logprobs;   // 0..20, with kFlagLogprobs
  uint32_t prefix_hint;      // prompt tokens the front-end expects to match the session (advisory)
  uint32_t n_stop_ids;       // ids that end generation (EOS is in here)
  uint32_t n_prompt;         // >= 1
  uint32_t reserved;
};

struct Cancel {
  uint64_t req_id;
};

struct Release {
  uint64_t session_id;
};

struct HelloEngine {
  uint32_t version;
  uint32_t max_context;   // prompt + output tokens per request
  uint32_t vocab_size;
  uint32_t eos_id;
  uint32_t max_sessions;  // resident sessions rank 0 can hold (advisory)
  uint32_t name_len;      // bytes of model/engine name that follow
};

enum TokensFlags : uint32_t {
  kTokLogprobs = 1,  // float logprob[n] follows the ids
  kTokTop = 2,       // TopLogprob top[n * n_top_logprobs] follows
};

struct TokensHeader {
  uint64_t req_id;
  uint32_t n;
  uint32_t flags;  // TokensFlags
};

struct TopLogprob {
  uint32_t id;
  float logprob;
};

enum DoneReason : uint32_t {
  kStop = 0,       // sampled a stop id (sent as the last token)
  kLength = 1,     // max_new_tokens reached
  kCancelled = 2,  // CANCEL
  kContext = 3,    // context window full
  kError = 4,      // rejected or failed; error text follows
};

struct DoneHeader {
  uint64_t req_id;
  uint32_t reason;             // DoneReason
  uint32_t prompt_tokens;
  uint32_t cached_tokens;      // prompt tokens whose state was reused
  uint32_t completion_tokens;  // ids sent in TOKENS for this request
  uint64_t queue_us;           // SUBMIT received -> prefill start
  uint64_t prefill_us;         // prefill start -> first token
  uint64_t decode_us;          // first token -> DONE
  uint32_t error_len;
  uint32_t reserved;
};

static_assert(sizeof(FrameHeader) == 8);
static_assert(sizeof(HelloFe) == 8);
static_assert(sizeof(SubmitHeader) == 72);
static_assert(offsetof(SubmitHeader, temperature) == 32);
static_assert(offsetof(SubmitHeader, n_prompt) == 64);
static_assert(sizeof(HelloEngine) == 24);
static_assert(sizeof(TokensHeader) == 16);
static_assert(sizeof(TopLogprob) == 8);
static_assert(sizeof(DoneHeader) == 56);
static_assert(offsetof(DoneHeader, queue_us) == 24);

}  // namespace fe
