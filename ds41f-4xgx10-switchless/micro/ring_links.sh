#!/bin/bash
# RoCE v2 GIDs of every RDMA device on this box: device, port state and rate, GID index, netdev, IPv4.
# Run on each box (e.g. ssh <box> bash -s < micro/ring_links.sh); docs/ring.md has the result.
h=$(hostname)
for d in /sys/class/infiniband/*; do
  dev=$(basename $d)
  for p in $d/ports/*; do
    port=$(basename $p)
    state=$(cat $p/state 2>/dev/null | awk '{print $2}')
    rate=$(cat $p/rate 2>/dev/null)
    for g in $p/gids/*; do
      i=$(basename $g)
      gid=$(cat $g 2>/dev/null)
      [ "$gid" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && continue
      typ=$(cat $p/gid_attrs/types/$i 2>/dev/null)
      nd=$(cat $p/gid_attrs/ndevs/$i 2>/dev/null)
      case "$gid" in 0000:0000:0000:0000:0000:ffff:*) ;; *) continue;; esac
      [ "$typ" = "RoCE v2" ] || continue
      ip=$(ip -4 -br addr show dev $nd 2>/dev/null | awk '{print $3}')
      echo "$h $dev port=$port $state rate=[$rate] gid_idx=$i netdev=$nd ip=$ip"
    done
  done
done
