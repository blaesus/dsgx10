#include "server/vision_files.h"
#include "server/vision_window.h"
#include "model/vision_layout.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sys/wait.h>

namespace fs = std::filesystem;
struct Span { int start, rows, offset; };
int checks = 0;
void check(bool yes) { assert(yes); ++checks; }
fe::ImageSpan image(uint32_t start, uint32_t rows, uint8_t id = 1) {
  return {start, rows, {id}};
}
int main() {
  check(fe::kVersion == 5);
  std::string gen, name;
  const std::string root = srv::kVisionFileRoot;
  check(srv::vision_file_path(root + "/good_gen-123/blob_123", gen, name));
  for (const auto& bad : std::vector<std::string>{
      "/etc/passwd", root + "/../passwd", root + "/a/../b", root + "/a/b/c",
      root + "//b", root + "/a/", root + "/a/b.bin", root + "/a/" + std::string(129,'x'),
      root + "/a/" + std::string("x\0y",3), root + "/a/中文"})
    check(!srv::vision_file_path(bad, gen, name));
  const bool made_root = ::mkdir(root.c_str(),0700) == 0;
  struct stat st{};
  check(!::lstat(root.c_str(), &st) && S_ISDIR(st.st_mode) &&
        (st.st_mode & 07777) == 0700 && st.st_uid == geteuid());
  const std::string owned_gen = root + "/cpu_test_" + std::to_string(getpid());
  check(::mkdir(owned_gen.c_str(),0700) == 0);
  struct Cleanup {
    std::string path, root; bool made_root;
    ~Cleanup() {
      // Exact fresh test generation only; never remove other frontend generations.
      assert(path.rfind(root + "/cpu_test_",0)==0);
      fs::remove_all(path); if(made_root) ::rmdir(root.c_str());
    }
  } cleanup{owned_gen,root,made_root};
  auto make = [&](const std::string& leaf, int rows, uint16_t value = 0x3f80) {
    const std::string path = owned_gen + "/" + leaf;
    const int fd = ::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);
    check(fd >= 0);
    std::vector<uint16_t> data(size_t(rows)*5120,value);
    check(::write(fd,data.data(),data.size()*2)==ssize_t(data.size()*2));
    check(!::fsync(fd) && !::fchmod(fd,0400)); ::close(fd);
    return path;
  };
  std::string error;
  const auto good = make("good",1024);
  auto data = srv::VisionFile::open(image(0,1024),good,error);
  check(bool(data) && error.empty() && srv::VisionFile::live_descriptors()==1);
  std::vector<uint16_t> payload(size_t(1024)*5120);
  check(data->read(payload.data(),[]{return false;},error));
  check(std::all_of(payload.begin(),payload.end(),[](uint16_t x){return x==0x3f80;}));
  auto copy = data;
  data.reset();
  check(srv::VisionFile::live_descriptors()==1);
  int cancel_calls = 0;
  error.clear();
  check(!copy->read(payload.data(),[&]{return ++cancel_calls>=3;},error));
  check(error=="image feature read cancelled");
  check(!::unlink(good.c_str()));
  error.clear();
  check(copy->read(payload.data(),[]{return false;},error));  // pinned inode survives frontend unlink
  copy.reset(); check(srv::VisionFile::live_descriptors()==0);
  auto reject = [&](const fe::ImageSpan& desc,const std::string& path) {
    error.clear(); auto p=srv::VisionFile::open(desc,path,error);
    check(!p && !error.empty() && srv::VisionFile::live_descriptors()==0);
  };
  reject(image(0,1),owned_gen+"/missing");
  const auto wrong_size = make("wrong_size",2);
  reject(image(0,1),wrong_size);
  reject(image(0,0),wrong_size);
  reject(image(0,1025),wrong_size);
  check(!::chmod(wrong_size.c_str(),0600));
  reject(image(0,2),wrong_size);
  check(!::chmod(wrong_size.c_str(),0400));
  const auto symlink = owned_gen+"/sym";
  check(!::symlink(wrong_size.c_str(),symlink.c_str()));
  reject(image(0,2),symlink);
  const auto hardlink = owned_gen+"/hard";
  check(!::link(wrong_size.c_str(),hardlink.c_str()));
  reject(image(0,2),wrong_size);
  check(!::unlink(hardlink.c_str()));
  check(!::mkfifo((owned_gen+"/fifo").c_str(),0400));
  reject(image(0,2),owned_gen+"/fifo"); // O_NONBLOCK prevents waiting for a writer
  check(!::chmod(owned_gen.c_str(),0755));
  reject(image(0,2),wrong_size);
  check(!::chmod(owned_gen.c_str(),0700));
  const auto nonfinite = make("nonfinite",1,0x7f80);
  auto nf = srv::VisionFile::open(image(0,1),nonfinite,error);
  check(bool(nf)); error.clear();
  check(!nf->read(payload.data(),[]{return false;},error) && error=="non-finite image feature file");
  nf.reset();
  auto changed = srv::VisionFile::open(image(0,2),wrong_size,error); check(bool(changed));
  check(!::chmod(wrong_size.c_str(),0600));
  const int mutate = ::open(wrong_size.c_str(),O_WRONLY|O_CLOEXEC);
  check(mutate>=0 && !::ftruncate(mutate,1) && !::fchmod(mutate,0400)); ::close(mutate);
  error.clear();
  check(!changed->read(payload.data(),[]{return false;},error) && error=="image feature file changed after submit");
  changed.reset(); check(srv::VisionFile::live_descriptors()==0);
  // Lowering the soft fd limit must reject before open, with no leaked reservations.
  const pid_t child=fork(); check(child>=0);
  if(child==0) {
    struct rlimit r{}; if(getrlimit(RLIMIT_NOFILE,&r)) _exit(2);
    r.rlim_cur=128; if(setrlimit(RLIMIT_NOFILE,&r)) _exit(3);
    std::string why;
    auto p=srv::VisionFile::open(image(0,1),nonfinite,why);
    _exit(!p && why=="image feature file descriptor budget exceeded" &&
          srv::VisionFile::live_descriptors()==0 ? 0:4);
  }
  int status=0; check(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);

  // Actual 58-distinct-image geometry exceeds 512 MiB logically but each 8192-row window is bounded.
  std::vector<fe::ImageSpan> images;
  std::vector<Span> layout;
  int image_start=0;
  uint64_t actual_rows=0;
  for(int i=0;i<58;++i) {
    const int rows=i<49?962:i<54?822:542;
    images.push_back(image(image_start,rows,uint8_t(i)));
    layout.push_back({image_start,rows,0});
    image_start+=rows+7; actual_rows+=rows;
  }
  constexpr int ctx=1<<20;
  check(actual_rows==53416 && actual_rows*5120*2==546979840);
  check(actual_rows*5120*2 > model::kMaxVisionFeatureBytes);
  check(model::detail::valid_vision_layout(layout,ctx));
  std::vector<uint8_t> live;
  model::detail::vision_live_layout(live,layout,ctx);
  std::vector<int> seen(58);
  for(int begin=0;begin<61000;begin+=8192) {
    std::vector<size_t> selected; uint64_t bytes=0; error.clear();
    const int count=std::min(8192,61000-begin);
    check(srv::vision_window(images,begin,count,model::kMaxVisionFeatureBytes,selected,bytes,error));
    check(bytes <= uint64_t(count+2046)*5120*2);
    std::vector<Span> chunk; int offset=0;
    for(size_t index:selected) {
      ++seen[index]; chunk.push_back({int(images[index].start),int(images[index].rows),offset});
      offset+=images[index].rows;
    }
    check(model::detail::valid_vision_chunk(chunk,offset,ctx));
    for(int pos=begin;pos<begin+count;++pos) {
      bool is_image=false;
      for(const auto& span:chunk) if(pos>=span.start && pos<span.start+span.rows)is_image=true;
      check(is_image==(live[pos]==0)); // no stale or missing per-chunk router/Engram image mask
    }
  }
  check(std::all_of(seen.begin(),seen.end(),[](int n){return n>0;}));
  check(std::any_of(seen.begin(),seen.end(),[](int n){return n==2;})); // whole image across boundary
  // Last 31 text rows: a mask query may cover an earlier chunk independently of features.
  std::vector<fe::ImageSpan> tail_images{image(7000,1024)};
  std::vector<Span> tail_layout{{7000,1024,0}};
  model::detail::vision_live_layout(live,tail_layout,ctx);
  std::vector<size_t> selected; uint64_t bytes=0;
  check(srv::vision_window(tail_images,8192,31,model::kMaxVisionFeatureBytes,selected,bytes,error));
  check(selected.empty() && bytes==0);
  int tail_mask_count=0;
  for(int pos=8223-2560;pos<8223;++pos) tail_mask_count+=live[pos]==0;
  check(tail_mask_count==1024); // General full-layout mask query; not an enabled CED replay policy
  // q lies immediately after image end: no features, but previous three Engram positions remain image.
  check(srv::vision_window(tail_images,8024,2,model::kMaxVisionFeatureBytes,selected,bytes,error));
  check(selected.empty() && live[8021]==0 && live[8022]==0 && live[8023]==0 && live[8024]==1);
  check(!srv::vision_window(images,0,61000,model::kMaxVisionFeatureBytes,selected,bytes,error));
  check(!srv::vision_window(images,-1,8192,model::kMaxVisionFeatureBytes,selected,bytes,error));
  check(!srv::vision_window(images,0,0,model::kMaxVisionFeatureBytes,selected,bytes,error));
  auto overlap=layout; overlap[1].start=1;
  check(!model::detail::valid_vision_layout(overlap,ctx));
  auto bad_offset=layout; bad_offset[0].offset=1;
  check(!model::detail::valid_vision_layout(bad_offset,ctx));
  check(!model::detail::valid_vision_layout(std::vector<Span>{{ctx-1,2,0}},ctx));
  check(!model::detail::valid_vision_layout(std::vector<Span>(4097,{0,1,0}),ctx));
  check(!model::detail::valid_vision_chunk(std::vector<Span>{{0,2,0}},1,ctx));
  check(!model::detail::valid_vision_chunk(std::vector<Span>{{0,1,1}},1,ctx));
  model::detail::vision_live_layout(live,std::vector<Span>{},ctx);
  check(live.empty());
  std::cout<<"VISION_BUDGET_CPU_PASS assertions="<<checks
           <<" max_fd="<<srv::VisionFile::descriptor_limit()
           <<" protocol=5 full_history_images=58 logical_bytes=546979840\n";
}
