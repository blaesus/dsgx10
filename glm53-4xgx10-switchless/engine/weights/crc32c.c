/* CRC32C (Castagnoli) with the ARMv8 CRC instructions, as io.h's crc32c, for glw.py (ctypes). Chainable:
   glw_crc32c(b, n, glw_crc32c(a, m, 0)) is the CRC of a followed by b. */
#include <arm_acle.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
#endif
__attribute__((target("+crc"))) uint32_t glw_crc32c(const void* data, size_t n, uint32_t crc) {
  const uint8_t* p = (const uint8_t*)data;
  crc = ~crc;
  while (n && ((uintptr_t)p & 7)) {
    crc = __crc32cb(crc, *p++);
    n--;
  }
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    crc = __crc32cd(crc, v);
    p += 8;
    n -= 8;
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return ~crc;
}
