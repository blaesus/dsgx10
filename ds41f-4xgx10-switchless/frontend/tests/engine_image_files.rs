//! Protocol 5 uses immutable feature-file leases; ImageMiss is no longer a retry path.
use dsfront::{engine::{EngineClient,Event},feature_file::FeatureFile,protocol::*};
use tokio::{net::UnixListener,io::AsyncWriteExt};
use std::{path::Path,time::Duration};
async fn receive(rd:&mut tokio::net::UnixStream)->Frame {
    let (ty,b)=read_frame(rd).await.unwrap().unwrap();Frame::decode(ty,&b,|_|0).unwrap()
}
async fn hello(stream:&mut tokio::net::UnixStream) {
    assert_eq!(receive(stream).await,Frame::Hello{version:5});
    stream.write_all(&Frame::HelloEngine(HelloEngine {version:VERSION,max_context:1048576,vocab_size:129280,eos_id:1,max_sessions:256,name:"isolated-cpu-files".into()}).encode()).await.unwrap();
}
fn request(digests:&[u8])->Submit {
    Submit {req_id:0,session_id:777,base_session_id:0,seed:1,temperature:0.,top_p:1.,top_k:0,max_new_tokens:8,
        flags:FLAG_KEEP,n_top_logprobs:0,prefix_hint:0,stop_ids:vec![1],prompt:vec![129264;digests.len()+1],
        images:digests.iter().enumerate().map(|(i,&d)|ImageSpan {start:i as u32,rows:1,digest:[d;32],
            feature:FeatureFile::create(&vec![0;5120*2]).unwrap()}).collect()}
}
fn done(s:&Submit,reason:DoneReason)->Frame {
    Frame::Done(Done {req_id:s.req_id,reason,prompt_tokens:s.prompt.len() as u32,cached_tokens:0,
        completion_tokens:0,queue_us:0,prefill_us:0,decode_us:0,error:String::new()})
}
async fn all_deleted(paths:&[String]) {
    for _ in 0..100 {
        if paths.iter().all(|p|!Path::new(p).exists()){return;}
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("feature owner leaked after terminal event");
}
#[tokio::test]
async fn files_remain_owned_through_cancel_until_terminal_done() {
    let path=std::env::temp_dir().join(format!("engine-files-cancel-{}.sock",std::process::id()));
    let listener=UnixListener::bind(&path).unwrap();
    let worker=tokio::spawn(async move {
        let (mut stream,_)=listener.accept().await.unwrap();hello(&mut stream).await;
        let Frame::Submit(s)=receive(&mut stream).await else {panic!("submit expected")};
        assert_eq!(s.images.len(),2);
        for image in &s.images { assert_eq!(std::fs::read(image.feature.path()).unwrap(),vec![0;5120*2]); }
        assert_eq!(receive(&mut stream).await,Frame::Cancel{req_id:s.req_id});
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(s.images.iter().all(|i|Path::new(i.feature.path()).exists()));
        stream.write_all(&done(&s,DoneReason::Cancelled).encode()).await.unwrap();
    });
    let client=EngineClient::start(path.clone());assert!(client.wait_connected(Duration::from_secs(3)).await.is_some());
    let s=request(&[1,2]);let paths:Vec<_>=s.images.iter().map(|i|i.feature.path().to_owned()).collect();
    let (id,mut events)=client.submit(s).unwrap();client.cancel(id);
    let Some(Event::Done(d))=tokio::time::timeout(Duration::from_secs(3),events.recv()).await.unwrap() else {panic!("Done expected")};
    assert_eq!(d.reason,DoneReason::Cancelled);
    worker.await.unwrap();all_deleted(&paths).await;std::fs::remove_file(path).unwrap();
}
#[tokio::test]
async fn disconnect_drains_all_pending_file_owners_without_resubmit() {
    let path=std::env::temp_dir().join(format!("engine-files-disconnect-{}.sock",std::process::id()));
    let listener=UnixListener::bind(&path).unwrap();
    let worker=tokio::spawn(async move {
        let (mut stream,_)=listener.accept().await.unwrap();hello(&mut stream).await;
        assert!(matches!(receive(&mut stream).await,Frame::Submit(_)));
        // Writer may still have more queued descriptors. Connection failure must release all owners.
        drop(stream);
    });
    let c=EngineClient::start(path.clone());assert!(c.wait_connected(Duration::from_secs(3)).await.is_some());
    let mut paths=Vec::new();let mut receivers=Vec::new();
    // No await between submissions, so they enter the same current connection.
    for n in 0..4 {
        let s=request(&[n]);paths.push(s.images[0].feature.path().to_owned());
        receivers.push(c.submit(s).unwrap().1);
    }
    for mut events in receivers {
        let Some(Event::Done(d))=tokio::time::timeout(Duration::from_secs(3),events.recv()).await.unwrap() else {panic!("done")};
        assert_eq!(d.reason,DoneReason::Error);
    }
    worker.await.unwrap();all_deleted(&paths).await;std::fs::remove_file(path).unwrap();
}
#[tokio::test]
async fn image_miss_is_protocol_error_and_never_resends_payload() {
    for unknown in [false,true] {
        let path=std::env::temp_dir().join(format!("engine-files-miss-{}-{unknown}.sock",std::process::id()));
        let listener=UnixListener::bind(&path).unwrap();
        let worker=tokio::spawn(async move {
            let (mut stream,_)=listener.accept().await.unwrap();hello(&mut stream).await;
            let Frame::Submit(s)=receive(&mut stream).await else {panic!("submit expected")};
            stream.write_all(&Frame::ImageMiss{req_id:s.req_id+if unknown {99}else{0},indices:vec![0]}.encode()).await.unwrap();
            assert!(tokio::time::timeout(Duration::from_secs(3),read_frame(&mut stream)).await.unwrap().unwrap().is_none());
        });
        let c=EngineClient::start(path.clone());assert!(c.wait_connected(Duration::from_secs(3)).await.is_some());
        let s=request(&[7]);let paths=vec![s.images[0].feature.path().to_owned()];
        let (_,mut events)=c.submit(s).unwrap();
        let Some(Event::Done(d))=tokio::time::timeout(Duration::from_secs(3),events.recv()).await.unwrap() else {panic!("done")};
        assert_eq!(d.reason,DoneReason::Error);
        worker.await.unwrap();all_deleted(&paths).await;std::fs::remove_file(path).unwrap();
    }
}
