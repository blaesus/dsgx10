#!/bin/bash
# RDMA write bandwidth on one link, one PCI domain vs both at once (perftest). The servers bind
# their TCP handshake to the link address (--bind_source_ip), never to 0.0.0.0.
for S in 1048576 8388608; do
  echo "## L1 rank 1 -> rank 2, $S B, domain 0 only"
  ssh 10.0.0.3 "setsid nohup timeout 30 ib_write_bw -d rocep1s0f1 --bind_source_ip 192.168.0.2 -x 3 -s $S -D 3 -F --report_gbits -p 18601 > /dev/null 2>&1 < /dev/null &"
  sleep 1; ssh 10.0.0.4 "ib_write_bw -d rocep1s0f0 -x 3 -s $S -D 3 -F --report_gbits -p 18601 192.168.0.2" 2>&1 | grep -E "^ *$S" | awk '{print "  BW avg Gb/s:", $4}'
  sleep 2
  echo "## L1 rank 1 -> rank 2, $S B, both domains"
  ssh 10.0.0.3 "setsid nohup timeout 30 ib_write_bw -d rocep1s0f1 --bind_source_ip 192.168.0.2 -x 3 -s $S -D 3 -F --report_gbits -p 18601 > /dev/null 2>&1 < /dev/null &"
  ssh 10.0.0.3 "setsid nohup timeout 30 ib_write_bw -d roceP2p1s0f1 --bind_source_ip 192.168.1.2 -x 3 -s $S -D 3 -F --report_gbits -p 18602 > /dev/null 2>&1 < /dev/null &"
  sleep 1
  ssh 10.0.0.4 "ib_write_bw -d rocep1s0f0 -x 3 -s $S -D 3 -F --report_gbits -p 18601 192.168.0.2 > /tmp/bw_d0.txt 2>&1 & ib_write_bw -d roceP2p1s0f0 -x 3 -s $S -D 3 -F --report_gbits -p 18602 192.168.1.2 > /tmp/bw_d2.txt 2>&1; wait; grep -hE '^ *$S' /tmp/bw_d0.txt /tmp/bw_d2.txt" | awk '{s+=$4; print "  domain BW avg Gb/s:", $4} END {print "  total:", s}'
  sleep 2
  echo "## L0 rank 0 -> rank 1 (100G), $S B, domain 0 only"
  ssh 10.0.0.4 "setsid nohup timeout 30 ib_write_bw -d rocep1s0f1 --bind_source_ip 192.168.6.2 -x 3 -s $S -D 3 -F --report_gbits -p 18601 > /dev/null 2>&1 < /dev/null &"
  sleep 1; ib_write_bw -d rocep1s0f0 -x 3 -s $S -D 3 -F --report_gbits -p 18601 192.168.6.2 2>&1 | grep -E "^ *$S" | awk '{print "  BW avg Gb/s:", $4}'
  sleep 2
  echo "## L0 rank 0 -> rank 1 (100G), $S B, both domains"
  ssh 10.0.0.4 "setsid nohup timeout 30 ib_write_bw -d rocep1s0f1 --bind_source_ip 192.168.6.2 -x 3 -s $S -D 3 -F --report_gbits -p 18601 > /dev/null 2>&1 < /dev/null &"
  ssh 10.0.0.4 "setsid nohup timeout 30 ib_write_bw -d roceP2p1s0f1 --bind_source_ip 192.168.7.2 -x 3 -s $S -D 3 -F --report_gbits -p 18602 > /dev/null 2>&1 < /dev/null &"
  sleep 1
  ib_write_bw -d rocep1s0f0 -x 3 -s $S -D 3 -F --report_gbits -p 18601 192.168.6.2 > /tmp/bw_d0.txt 2>&1 &
  ib_write_bw -d roceP2p1s0f0 -x 3 -s $S -D 3 -F --report_gbits -p 18602 192.168.7.2 > /tmp/bw_d2.txt 2>&1
  wait
  grep -hE "^ *$S" /tmp/bw_d0.txt /tmp/bw_d2.txt | awk '{s+=$4; print "  domain BW avg Gb/s:", $4} END {print "  total:", s}'
  sleep 2
done
