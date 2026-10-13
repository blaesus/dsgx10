// dequery: does this GPU have the copy engine's hardware decompression? A driver query only:
// cuInit and device attributes, no context, no memory, no kernels (like nvidia-smi).
#include <cuda.h>
#include <stdio.h>

int main(void) {
  CUdevice d;
  int mask = -1, maxlen = -1, sms = -1;
  char name[256];
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&d, 0) != CUDA_SUCCESS) return 1;
  cuDeviceGetName(name, sizeof name, d);
  CUresult r1 = cuDeviceGetAttribute(&mask, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, d);
  CUresult r2 = cuDeviceGetAttribute(&maxlen, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH, d);
  cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d);
  printf("%s: SMs %d; decompress algorithm mask %d (r=%d; 1 deflate, 2 snappy, 4 lz4), max length %d (r=%d)\n",
         name, sms, mask, (int)r1, maxlen, (int)r2);
  return 0;
}
