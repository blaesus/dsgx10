//! CPU integration: real tokenizer/session planner/cache and Unix wire, synthetic
//! embeddings only. This does NOT qualify GPU numerical or visual correctness.
use dsfront::{vision::{self,FeatureCache,IMAGE_ID},tokenizer::Tokenizer};
use serde_json::{json,Value};
use std::sync::{Arc,Mutex};
use tokio::{net::UnixListener,io::{AsyncReadExt,AsyncWriteExt}};

#[tokio::test]
#[ignore="requires DSGB10_TEST_TOKENIZER and >=700 MiB disk spool; isolated CPU only"]
async fn fifty_eight_distinct_full_resolution_images_are_batched_spooled_and_reused() {
    let tok=Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
    let path=std::env::temp_dir().join(format!("vision-history58-cpu-{}.sock",std::process::id()));
    let listener=UnixListener::bind(&path).unwrap();
    let counts=Arc::new(Mutex::new(Vec::<usize>::new()));let seen=counts.clone();
    let task=tokio::spawn(async move {
        loop {
            let (mut stream,_)=listener.accept().await.unwrap();
            let n=stream.read_u32_le().await.unwrap();let mut b=vec![0;n as usize];stream.read_exact(&mut b).await.unwrap();
            let body:Value=serde_json::from_slice(&b).unwrap();
            let images=body.get("images").and_then(Value::as_array);
            if let Some(images)=images { assert!(images.len()<=8);seen.lock().unwrap().push(images.len()); }
            let meta:Vec<_>=images.into_iter().flatten().map(|r| {
                let id=r["data"].as_str().unwrap().parse::<u64>().unwrap();
                json!({"rows":1024,"dim":5120,"digest":format!("{id:064x}")})
            }).collect();
            let header=json!({"ok":true,"revision":"test-rev","identity":"11".repeat(32),"images":meta});
            let b=serde_json::to_vec(&header).unwrap();stream.write_u32_le(b.len() as u32).await.unwrap();stream.write_all(&b).await.unwrap();
            for _ in &meta { stream.write_all(&vec![0;1024*5120*2]).await.unwrap(); }
        }
    });
    let mut cache=FeatureCache::default();
    let records:Vec<_>=(0..58).map(|i|json!({"data":i.to_string()})).collect();
    let raw=(0..58).map(|i|format!("Turn {i}: <｜deepseek_image｜>\n")).collect::<String>();
    let a=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&raw,"normal",&tok,1).await.unwrap();
    assert_eq!(a.images.len(),58);
    assert_eq!(a.images.iter().map(|i|i.feature.len()).sum::<usize>(),580<<20);
    assert_eq!(*counts.lock().unwrap(),vec![8,8,8,8,8,8,8,2]);
    let ids=tok.encode(&a.prompt);
    for (index,image) in a.images.iter().enumerate() {
        assert_eq!(std::fs::metadata(image.feature.path()).unwrap().len(),10<<20);
        assert_eq!(image.digest[31],index as u8);
        assert!(ids[image.start as usize..(image.start+image.rows) as usize].iter().all(|&id|id==IMAGE_ID));
    }
    let b=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&raw,"normal",&tok,1).await.unwrap();
    assert_eq!(counts.lock().unwrap().len(),8,"unchanged history must not re-encode");
    assert!(a.images.iter().zip(&b.images).all(|(a,b)|a.feature.shares_owner(&b.feature)));
    let mut appended=records.clone();appended.extend((58..62).map(|i|json!({"data":i.to_string()})));
    let more=format!("{raw}{}", "<｜deepseek_image｜>\n".repeat(4));
    let c=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&appended,&more,"normal",&tok,1).await.unwrap();
    assert_eq!(counts.lock().unwrap().last().copied(),Some(4));
    assert_eq!(c.images.len(),62);
    let paths:Vec<_>=c.images.iter().map(|i|i.feature.path().to_owned()).collect();
    drop(cache);drop(a);drop(b);
    assert!(paths.iter().all(|p|std::path::Path::new(p).exists()),"inflight owners outlive cache");
    drop(c);
    assert!(paths.iter().all(|p|!std::path::Path::new(p).exists()));
    task.abort();let _=task.await;std::fs::remove_file(path).unwrap();
}
