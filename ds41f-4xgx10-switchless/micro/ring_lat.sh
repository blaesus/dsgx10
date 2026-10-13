#!/bin/bash
# One-way latency per ring link (domain 0): CPU ping-pong and GPU ping-pong through a CPU proxy.
# Needs micro/rdma_lat (nvcc -O3 -arch=sm_121a rdma_lat.cu -libverbs) in ~/dsgb10/bin on all boxes.
B=dsgb10/bin/rdma_lat
for S in 64 10240 61440; do
  echo "## L0 rank 0 (rocep1s0f0) <-> rank 1 (rocep1s0f1) $S B"
  ssh 10.0.0.4 "setsid nohup ~/$B server 192.168.6.2 rocep1s0f1 3 $S > ~/dsgb10/logs/lat_srv.log 2>&1 < /dev/null &"
  sleep 1; timeout 60 ~/dsgb10/bin/rdma_lat client 192.168.6.2 rocep1s0f0 3 $S 2>&1 | grep -o "one-way.*"
  sleep 1
  echo "## L1 rank 1 (rocep1s0f0) <-> rank 2 (rocep1s0f1) $S B"
  ssh 10.0.0.3 "setsid nohup ~/$B server 192.168.0.2 rocep1s0f1 3 $S > ~/dsgb10/logs/lat_srv.log 2>&1 < /dev/null &"
  sleep 1; ssh 10.0.0.4 "timeout 60 ~/$B client 192.168.0.2 rocep1s0f0 3 $S" 2>&1 | grep -o "one-way.*"
  sleep 1
  echo "## L2 rank 2 (rocep1s0f0) <-> rank 3 (rocep1s0f1) $S B"
  ssh 10.0.0.2 "setsid nohup ~/$B server 192.168.4.2 rocep1s0f1 3 $S > ~/dsgb10/logs/lat_srv.log 2>&1 < /dev/null &"
  sleep 1; ssh 10.0.0.3 "timeout 60 ~/$B client 192.168.4.2 rocep1s0f0 3 $S" 2>&1 | grep -o "one-way.*"
  sleep 1
  echo "## L3 rank 3 (rocep1s0f0) <-> rank 0 (rocep1s0f1) $S B"
  ssh 10.0.0.2 "setsid nohup ~/$B server 192.168.2.2 rocep1s0f0 3 $S > ~/dsgb10/logs/lat_srv.log 2>&1 < /dev/null &"
  sleep 1; timeout 60 ~/dsgb10/bin/rdma_lat client 192.168.2.2 rocep1s0f1 3 $S 2>&1 | grep -o "one-way.*"
  sleep 1
done
