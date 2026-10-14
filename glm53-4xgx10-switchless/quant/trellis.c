/* Tail-biting bitshift-trellis Viterbi quantiser for EXL3-style 16-bit-state codes (CPU).
 *
 * A tile is a ring of L weights. Position i carries a 16-bit word; its top 16-K bits equal the previous
 * position's low 16-K bits (the trellis edge), its low K bits are new (the branch), and the word decodes
 * through a 65,536-entry codebook cb[]. The search is exllamav3's (quantize_tiles_kernel.cuh): a first
 * unconstrained pass over the ring rotated by L/2, traced back to position 0 to fix the wrap-around edge,
 * then a second pass constrained to start from that edge and traced back from it. Differences from the
 * CUDA kernel: costs and inputs are f32 (the kernel converts the tile to f16 and keeps f16 costs), and
 * ties go to the smallest branch value and the smallest final edge. Compiled without fast-math and with
 * -ffp-contract=off, so the result is a pure function of the inputs; tiles are independent, so the thread
 * count cannot change any output.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>

/* One trellis step: best[e] = min over the K-bit branch k of prev[in_edge(k, e)] + (cb[word(k, e)] - w)^2,
 * word(k, e) = k << (16 - K) | e, in_edge = word >> K; bk[e] = the minimising k (smallest on ties). With
 * prev == NULL the previous costs are 0, or +inf except at in_edge == pre when pre >= 0. The Q outputs that
 * share e >> K are kept in registers across the Q branches; Q is a compile-time constant per K. */
#define STEP(KK)                                                                                      \
static void step##KK(const float *prev, int pre, float w, const float *cb, float *best, uint8_t *bk) \
{                                                                                                     \
    enum { K = KK, Kr = 16 - KK, Q = 1 << KK, NV = (1 << KK) / 4, H = (1 << (16 - KK)) >> KK };       \
    const float32x4_t wv = vdupq_n_f32(w);                                                            \
    for (int eh = 0; eh < H; eh++) {                                                                  \
        float32x4_t acc[NV]; uint32x4_t idx[NV];                                                      \
        for (int v = 0; v < NV; v++) { acc[v] = vdupq_n_f32(INFINITY); idx[v] = vdupq_n_u32(0); }     \
        for (int k = 0; k < Q; k++) {                                                                 \
            const int in_edge = (k << (Kr - K)) | eh;                                                 \
            float p;                                                                                  \
            if (prev) p = prev[in_edge];                                                              \
            else p = (pre < 0 || in_edge == pre) ? 0.0f : INFINITY;                                   \
            const float32x4_t pv = vdupq_n_f32(p);                                                    \
            const uint32x4_t kv = vdupq_n_u32((uint32_t)k);                                           \
            const float *c = cb + (k << Kr) + (eh << K);                                              \
            for (int v = 0; v < NV; v++) {                                                            \
                float32x4_t d = vsubq_f32(vld1q_f32(c + 4 * v), wv);                                  \
                float32x4_t x = vaddq_f32(pv, vmulq_f32(d, d));                                       \
                uint32x4_t lt = vcltq_f32(x, acc[v]);                                                 \
                acc[v] = vbslq_f32(lt, x, acc[v]);                                                    \
                idx[v] = vbslq_u32(lt, kv, idx[v]);                                                   \
            }                                                                                         \
        }                                                                                             \
        for (int v = 0; v < NV; v++) {                                                                \
            vst1q_f32(best + (eh << K) + 4 * v, acc[v]);                                              \
            uint32_t t[4]; vst1q_u32(t, idx[v]);                                                      \
            for (int j = 0; j < 4; j++) bk[(eh << K) + 4 * v + j] = (uint8_t)t[j];                    \
        }                                                                                             \
    }                                                                                                 \
}
STEP(2) STEP(3) STEP(4) STEP(5) STEP(6)

static void step(const float *prev, int pre, float w, const float *cb, int K, float *best, uint8_t *bk)
{
    switch (K) {
    case 2: step2(prev, pre, w, cb, best, bk); break;
    case 3: step3(prev, pre, w, cb, best, bk); break;
    case 4: step4(prev, pre, w, cb, best, bk); break;
    case 5: step5(prev, pre, w, cb, best, bk); break;
    default: step6(prev, pre, w, cb, best, bk); break;
    }
}

/* One pass over the ring starting at position roll; bp[ri * E + e] gets the branch of the best word whose
 * low bits are e at position ri. Returns the cost array of the last step (one of c0/c1). */
static float *pass(const float *w, int L, int K, const float *cb, int roll, int pre, float *c0, float *c1,
                   uint8_t *bp)
{
    const int E = 1 << (16 - K);
    float *cur = c0, *nxt = c1;
    for (int i = 0; i < L; i++) {
        int ri = (i + roll) % L;
        step(i ? cur : NULL, i ? -1 : pre, w[ri], cb, K, nxt, bp + (size_t)ri * E);
        float *t = cur; cur = nxt; nxt = t;
    }
    return cur;
}

static void tile(const float *w, int L, int K, const float *cb, uint16_t *words, float *vals, float *c0,
                 float *c1, uint8_t *bp)
{
    const int Kr = 16 - K, E = 1 << Kr;
    float *last = pass(w, L, K, cb, L / 2, -1, c0, c1, bp);
    int e = 0;
    for (int j = 1; j < E; j++) if (last[j] < last[e]) e = j;
    for (int i = L - 1; i >= 0; i--) {
        int ri = (i + L / 2) % L;
        int word = (bp[(size_t)ri * E + e] << Kr) | e;
        e = word >> K;
        if (ri == 0) break;
    }
    const int end = e;
    pass(w, L, K, cb, 0, end, c0, c1, bp);
    e = end;
    for (int i = L - 1; i >= 0; i--) {
        int word = (bp[(size_t)i * E + e] << Kr) | e;
        words[i] = (uint16_t)word;
        vals[i] = cb[word];
        e = word >> K;
    }
    /* e == end here: the ring closes (checked by the Python side on the words) */
}

/* tiles: T x L f32; cb: 65536 f32; out words T x L u16, vals T x L f32. Returns 0. */
int viterbi_tiles(const float *tiles, int T, int L, int K, /* 2 <= K <= 6 */ const float *cb, uint16_t *words, float *vals,
                  int nthreads)
{
    const int E = 1 << (16 - K);
    #pragma omp parallel num_threads(nthreads)
    {
        float *c0 = malloc(sizeof(float) * E), *c1 = malloc(sizeof(float) * E);
        uint8_t *bp = malloc((size_t)L * E);
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < T; t++)
            tile(tiles + (size_t)t * L, L, K, cb, words + (size_t)t * L, vals + (size_t)t * L, c0, c1, bp);
        free(c0); free(c1); free(bp);
    }
    return 0;
}
