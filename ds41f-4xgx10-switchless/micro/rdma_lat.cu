// RDMA latency between two neighbouring GB10s: CPU ping-pong and GPU ping-pong via CPU proxy.
// usage: rdma_lat <server|client> <tcp_ip> <ibdev> <gid_idx> [bytes]
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define DIE(...) do{fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); exit(1);}while(0)
struct Info { uint32_t qpn, psn, rkey; uint64_t addr; uint8_t gid[16]; };
static int tcp_connect(bool server, const char* ip, int port){
  int s = socket(AF_INET, SOCK_STREAM, 0); int one=1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, 4);
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET, ip, &a.sin_addr);
  if (server){ if(bind(s,(sockaddr*)&a,sizeof a)) DIE("bind"); listen(s,1); int c=accept(s,0,0); close(s); setsockopt(c,IPPROTO_TCP,TCP_NODELAY,&one,4); return c; }
  for(int i=0;i<300;i++){ if(!connect(s,(sockaddr*)&a,sizeof a)) { setsockopt(s,IPPROTO_TCP,TCP_NODELAY,&one,4); return s;} usleep(100000);} DIE("connect");
}
static void xchg(int fd, void* mine, void* theirs, size_t n){ if(write(fd,mine,n)!=(ssize_t)n) DIE("w"); size_t g=0; while(g<n){ ssize_t r=read(fd,(char*)theirs+g,n-g); if(r<=0) DIE("r"); g+=r; } }

// GPU side: wait for flag==seq in recv buf, then write payload+flag to send buf (for proxy).
__global__ void gpu_pingpong(volatile uint64_t* recv_flag, volatile uint64_t* send_flag, uint64_t* send_payload, int iters, bool starter, int words){
  for (int i = 1; i <= iters; i++) {
    if (!starter || i > 1) { if (threadIdx.x == 0) { while (*recv_flag < (uint64_t)(starter ? i-1 : i)) ; } __syncthreads(); }
    for (int w = threadIdx.x; w < words; w += blockDim.x) send_payload[w] = i;
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); *send_flag = i; }
  }
  if (starter && threadIdx.x == 0) { while (*recv_flag < (uint64_t)iters) ; }
}
int main(int argc, char** argv){
  if (argc < 5) DIE("usage");
  bool server = !strcmp(argv[1], "server"); const char* ip = argv[2]; const char* dev = argv[3]; int gid_idx = atoi(argv[4]);
  size_t bytes = argc > 5 ? atol(argv[5]) : 64;
  int ndev; ibv_device** list = ibv_get_device_list(&ndev); ibv_context* ctx = 0;
  for (int i=0;i<ndev;i++) if(!strcmp(ibv_get_device_name(list[i]),dev)) ctx = ibv_open_device(list[i]);
  if(!ctx) DIE("no dev");
  ibv_pd* pd = ibv_alloc_pd(ctx);
  // one host-pinned region: [send: payload | flag] [recv: payload | flag]
  size_t region = 2 * (bytes + 64) + 4096; char* buf;
  CK(cudaHostAlloc((void**)&buf, region, cudaHostAllocMapped)); memset(buf, 0, region);
  char* sendb = buf; char* recvb = buf + ((bytes + 64 + 4095) & ~4095ul);
  ibv_mr* mr = ibv_reg_mr(pd, buf, region, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
  if(!mr) DIE("reg_mr %s", strerror(errno));
  ibv_cq* cq = ibv_create_cq(ctx, 1024, 0, 0, 0);
  ibv_qp_init_attr qia{}; qia.send_cq=cq; qia.recv_cq=cq; qia.qp_type=IBV_QPT_RC; qia.cap.max_send_wr=512; qia.cap.max_recv_wr=16; qia.cap.max_send_sge=1; qia.cap.max_recv_sge=1; qia.cap.max_inline_data=256;
  ibv_qp* qp = ibv_create_qp(pd, &qia); if(!qp) DIE("qp");
  ibv_qp_attr at{}; at.qp_state=IBV_QPS_INIT; at.pkey_index=0; at.port_num=1; at.qp_access_flags=IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_LOCAL_WRITE;
  if(ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) DIE("init");
  Info me{}, peer{}; me.qpn=qp->qp_num; me.psn=1234; me.rkey=mr->rkey; me.addr=(uint64_t)recvb;
  ibv_gid gid; ibv_query_gid(ctx,1,gid_idx,&gid); memcpy(me.gid,gid.raw,16);
  int fd = tcp_connect(server, ip, 18515); xchg(fd,&me,&peer,sizeof me);
  memset(&at,0,sizeof at); at.qp_state=IBV_QPS_RTR; at.path_mtu=IBV_MTU_4096; at.dest_qp_num=peer.qpn; at.rq_psn=peer.psn; at.max_dest_rd_atomic=1; at.min_rnr_timer=12;
  at.ah_attr.is_global=1; memcpy(at.ah_attr.grh.dgid.raw,peer.gid,16); at.ah_attr.grh.sgid_index=gid_idx; at.ah_attr.grh.hop_limit=1; at.ah_attr.port_num=1;
  if(ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)) DIE("rtr");
  memset(&at,0,sizeof at); at.qp_state=IBV_QPS_RTS; at.timeout=14; at.retry_cnt=7; at.rnr_retry=7; at.sq_psn=me.psn; at.max_rd_atomic=1;
  if(ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)) DIE("rts");
  char c=0; xchg(fd,&c,&c,1);
  volatile uint64_t* sflag = (volatile uint64_t*)(sendb + bytes); volatile uint64_t* rflag = (volatile uint64_t*)(recvb + bytes);
  int unsignaled = 0;
  auto post = [&](){ // write payload+flag
    ibv_sge sg{ (uint64_t)sendb, (uint32_t)(bytes + 8), mr->lkey };
    ibv_send_wr wr{}, *bad; wr.opcode=IBV_WR_RDMA_WRITE; wr.sg_list=&sg; wr.num_sge=1; wr.wr.rdma.remote_addr=peer.addr; wr.wr.rdma.rkey=peer.rkey;
    wr.send_flags = (bytes + 8 <= 256 ? IBV_SEND_INLINE : 0);
    if (++unsignaled >= 64) { wr.send_flags |= IBV_SEND_SIGNALED; unsignaled = 0; }
    if (ibv_post_send(qp,&wr,&bad)) DIE("post");
    ibv_wc wc; while (ibv_poll_cq(cq,1,&wc) > 0) if (wc.status) DIE("wc %d", wc.status);
  };
  // ---- CPU ping-pong
  int iters = 20000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 1; i <= iters; i++) {
    if (server) { while (*rflag < (uint64_t)i); *sflag = i; post(); }
    else { *sflag = i; post(); while (*rflag < (uint64_t)i); }
  }
  double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
  printf("[%s] CPU ping-pong %zu B: one-way %.2f us\n", server?"srv":"cli", bytes, us/iters/2);
  xchg(fd,&c,&c,1); *rflag = 0; *sflag = 0; xchg(fd,&c,&c,1);
  // ---- GPU ping-pong via proxy thread
  std::atomic<bool> stop{false};
  std::thread proxy([&]{ uint64_t last = 0; while(!stop.load(std::memory_order_relaxed)) { uint64_t v = *sflag; if (v != last) { last = v; post(); } } });
  iters = 5000;
  uint64_t *d_rflag, *d_sflag, *d_payload;
  CK(cudaHostGetDevicePointer((void**)&d_rflag, (void*)rflag, 0)); CK(cudaHostGetDevicePointer((void**)&d_sflag, (void*)sflag, 0)); CK(cudaHostGetDevicePointer((void**)&d_payload, sendb, 0));
  gpu_pingpong<<<1,256>>>(d_rflag, d_sflag, d_payload, 1, !server, (int)(bytes/8)); // warmup handled by ordering below
  CK(cudaDeviceSynchronize());
  xchg(fd,&c,&c,1); *rflag = 0; xchg(fd,&c,&c,1);
  // reset counters both sides
  *sflag = 0; std::this_thread::sleep_for(std::chrono::milliseconds(50)); *rflag = 0; xchg(fd,&c,&c,1);
  t0 = std::chrono::steady_clock::now();
  gpu_pingpong<<<1,256>>>(d_rflag, d_sflag, d_payload, iters, !server, (int)(bytes/8));
  CK(cudaDeviceSynchronize());
  us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
  printf("[%s] GPU ping-pong via CPU proxy %zu B: one-way %.2f us\n", server?"srv":"cli", bytes, us/iters/2);
  stop = true; proxy.join();
  xchg(fd,&c,&c,1);
  return 0;
}
