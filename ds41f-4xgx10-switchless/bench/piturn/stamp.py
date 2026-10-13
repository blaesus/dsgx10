#!/usr/bin/env python3
"""Stamps each line of stdin with its arrival (time.time_ns()) and writes 'NS line' to stdout, flushed:
dsfront's stderr through it gives each "req N plan:" and "req N: end:" line a clock to 0.1 ms."""
import sys
import time

out = sys.stdout.buffer
for line in sys.stdin.buffer:
    out.write(b'%d %s' % (time.time_ns(), line))
    out.flush()
