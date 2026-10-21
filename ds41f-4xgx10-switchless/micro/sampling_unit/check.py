#!/usr/bin/env python3
"""Exhaustive regression of the actual unit() expression; no weights or service.

Default: host C++ compiler. --cuda: explicit, opt-in standalone GPU check.
The function body is extracted from sample.cu instead of duplicating the fix.
If it delegates to the shared sampler helper, that actual header is compiled.
"""
import argparse
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OLD = 'return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--nvcc', default='nvcc')
    args = parser.parse_args()
    source = (ROOT / 'engine/model/sample.cu').read_text(encoding='utf-8')
    match = re.search(r'float\s+unit\(uint64_t h\)\s*\{([^}]+)\}', source)
    if not match:
        raise SystemExit('Cannot find actual sampler unit(uint64_t h)')
    qualifier = '__host__ __device__' if args.cuda else ''
    common = '''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
static_assert(std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24, "requires IEEE binary32");
'''
    # The probability branch shares its real helper with the draft sampler.
    # Compile the actual wrapper and header; do not replace it with a copied
    # expected formula, which could pass while the runtime helper is broken.
    if 'spec_sampling::' in match.group(1):
        common += '#include "common/spec_sampling.h"\n'
    common += f'{qualifier} float actual(uint64_t h) {{ {match.group(1)} }}\n'
    common += f'{qualifier} float original(uint64_t h) {{ {OLD} }}\n'
    # Independent properties: open interval, finite transformed value, exact
    # extreme values, and equal representation count for every midpoint.
    common += f'{qualifier} unsigned errors(uint32_t i) {{\n' + '''
  const uint64_t h = uint64_t(i) << 40;
  const float u = actual(h);
  unsigned n = !(u > 0.f && u < 1.f);
  const float g = -logf(-logf(u));
  n += !(g > -INFINITY && g < INFINITY);
  n += actual(h | ((uint64_t(1) << 40) - 1)) != u;
  if (i == 0) n += u != 0x1p-24f;
  if (i == (1u << 24) - 1) n += u != 1.f - 0x1p-24f;
  if (i & 1) n += u != actual(uint64_t(i-1) << 40);
  if (i && !(i & 1)) n += !(u > actual(uint64_t(i-1) << 40));
  return n;
}
'''
    if args.cuda:
        common = '#include <cuda_runtime.h>\n' + common + '''
__global__ void check(unsigned long long* result) {
  for (uint32_t i = blockIdx.x*blockDim.x+threadIdx.x; i < (1u<<24);
       i += gridDim.x*blockDim.x) {
    atomicAdd(result, (unsigned long long)errors(i));
    if (!(original(uint64_t(i)<<40) < 1.f)) atomicAdd(result+1, 1ULL);
  }
}
int main() {
  unsigned long long *device, result[2] = {};
  if (cudaMalloc(&device, sizeof result) != cudaSuccess) return 2;
  if (cudaMemset(device, 0, sizeof result) != cudaSuccess) return 3;
  check<<<64,256>>>(device);
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(result, device, sizeof result, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  if (cudaFree(device) != cudaSuccess) return 6;
  printf("cases=16777216 new_errors=%llu original_endpoint_errors=%llu\\n", result[0], result[1]);
  return result[0] != 0 || result[1] != 1;
}
'''
    else:
        common += '''
int main() {
  unsigned long long result[2] = {};
  for (uint32_t i = 0; i < (1u<<24); ++i) {
    result[0] += errors(i);
    result[1] += !(original(uint64_t(i)<<40) < 1.f);
  }
  printf("cases=16777216 new_errors=%llu original_endpoint_errors=%llu\\n", result[0], result[1]);
  return result[0] != 0 || result[1] != 1;
}
'''
    with tempfile.TemporaryDirectory(prefix='sampling-unit-') as tmp:
        path = pathlib.Path(tmp)
        src = path / ('check.cu' if args.cuda else 'check.cpp')
        exe = path / 'check'
        src.write_text(common, encoding='utf-8')
        compiler = args.nvcc if args.cuda else args.cxx
        if not args.cuda and pathlib.Path(compiler).name.lower() in ('cl', 'cl.exe'):
            exe = path / 'check.exe'
            command = [compiler, '/nologo', '/O2', '/std:c++17', '/EHsc', '/utf-8',
                       '/I' + str(ROOT / 'engine'), str(src),
                       '/Fe:' + str(exe), '/Fo:' + str(path / 'check.obj')]
        else:
            command = [compiler, '-O2', '-std=c++17', '-I', str(ROOT / 'engine'),
                       str(src), '-o', str(exe)]
        subprocess.run(command, check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
