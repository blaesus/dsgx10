// pcache: how much of each file sits in the page cache, and (--drop) drop it (posix_fadvise DONTNEED: clean pages
// only, no root needed). On GB10 CUDA allocates from free pages only, so the page cache left by a copy of a rank file
// is memory the cache pool cannot have (docs/serving.md § 3).
//   cc -O2 -o pcache pcache.c && ./pcache [--drop] FILE...
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char** argv) {
  int drop = 0, rc = 0;
  const long pg = sysconf(_SC_PAGESIZE);
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--drop")) { drop = 1; continue; }
    const int fd = open(argv[i], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) { perror(argv[i]); rc = 1; continue; }
    double cached = 0;
    const size_t chunk = (size_t)1 << 30;
    unsigned char* vec = malloc(chunk / pg + 1);
    for (off_t off = 0; off < st.st_size; off += chunk) {
      const size_t n = (size_t)(st.st_size - off) < chunk ? (size_t)(st.st_size - off) : chunk;
      void* p = mmap(NULL, n, PROT_READ, MAP_SHARED, fd, off);
      if (p == MAP_FAILED) { perror("mmap"); rc = 1; break; }
      if (mincore(p, n, vec) == 0)
        for (size_t k = 0; k < (n + pg - 1) / pg; ++k) cached += (vec[k] & 1) ? pg : 0;
      munmap(p, n);
    }
    free(vec);
    printf("%s: %.2f of %.2f GiB cached", argv[i], cached / 1073741824.0, st.st_size / 1073741824.0);
    if (drop) {
      fdatasync(fd);
      const int e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      printf("; dropped%s", e ? " (failed)" : "");
    }
    printf("\n");
    close(fd);
  }
  return rc;
}
