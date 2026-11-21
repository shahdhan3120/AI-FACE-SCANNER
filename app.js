import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const firebaseConfig={apiKey:"AIzaSyBOfy1EniuYItmFl103AibJd1gag6q4SNo",authDomain:"ai-face-scanner-ea2b5.firebaseapp.com",projectId:"ai-face-scanner-ea2b5",storageBucket:"ai-face-scanner-ea2b5.firebasestorage.app",messagingSenderId:"502564498315",appId:"1:502564498315:web:c48b23ff8772700c501d8e"};
const ADMIN_UID="m3YVRkYRz1Z11OEBV5I9xvtkFot2";
firebase.initializeApp(firebaseConfig);
const auth=firebase.auth();
const $=id=>document.getElementById(id);
const video=$("video"),photo=$("photoPreview"),canvas=$("overlay"),ctx=canvas.getContext("2d");
let user=null,landmarker=null,drawingUtils=null,stream=null,raf=0,lastVideoTime=-1,lastFpsTime=0,frames=0,fps=0,scans=0,faceTotal=0,history=[],lastReading=null,mirror=true,facing="user",currentImageUrl=null,lastSnapshot=null,scanMode="idle",modelLoading=false;

const labels=[
 {key:"happy",name:"Happy-like",emoji:"😊",desc:"Smile-related facial movement cues"},
 {key:"sad",name:"Sad-like",emoji:"😢",desc:"Frown and inner-brow movement cues"},
 {key:"angry",name:"Angry-like",emoji:"😠",desc:"Brow-lowering and eye-tension cues"},
 {key:"surprised",name:"Surprised-like",emoji:"😲",desc:"Raised-brow and open-jaw cues"},
 {key:"fearful",name:"Fearful-like",emoji:"😨",desc:"Wide-eye and raised-brow cues"},
 {key:"disgusted",name:"Disgust-like",emoji:"🤢",desc:"Nose and upper-lip movement cues"},
 {key:"neutral",name:"Neutral-like",emoji:"😐",desc:"Few strong expression movement cues"}
];
const clamp=n=>Math.max(0,Math.min(1,n));
const avg=(a,b)=>(a+b)/2;
function setStatus(text,mode=""){ $("scanStatus").textContent="● "+text.toUpperCase();$("modeLabel").textContent=mode.toUpperCase()||"WAITING"; }
function setMessage(id,text,error=false){$(id).textContent=text;$(id).style.color=error?"#ff8eaa":"";}
async function loadModel(){
 if(landmarker)return landmarker;
 if(modelLoading) {while(modelLoading) await new Promise(r=>setTimeout(r,150));return landmarker;}
 modelLoading=true;setStatus("Loading AI model","Loading");
 try{
  const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker=await FaceLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",delegate:"GPU"},runningMode:"VIDEO",numFaces:4,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:false});
  drawingUtils=new DrawingUtils(ctx);setStatus("Model ready","Ready");return landmarker;
 }catch(e){console.error(e);setStatus("Model error","Error");throw new Error("Could not load the expression model. Check your internet connection and refresh.");}
 finally{modelLoading=false;}
}
function getBlendMap(result){const cats=result?.faceBlendshapes?.[0]?.categories||[];return Object.fromEntries(cats.map(c=>[c.categoryName,c.score]));}
function scoreExpressions(result){
 const b=getBlendMap(result),g=k=>b[k]||0;
 const smile=avg(g("mouthSmileLeft"),g("mouthSmileRight")),frown=avg(g("mouthFrownLeft"),g("mouthFrownRight"));
 const browDown=avg(g("browDownLeft"),g("browDownRight")),eyeWide=avg(g("eyeWideLeft"),g("eyeWideRight")),eyeSquint=avg(g("eyeSquintLeft"),g("eyeSquintRight"));
 const press=avg(g("mouthPressLeft"),g("mouthPressRight")),cheek=avg(g("cheekSquintLeft"),g("cheekSquintRight")),sneer=avg(g("noseSneerLeft"),g("noseSneerRight"));
 const inner=g("browInnerUp"),jaw=g("jawOpen"),pucker=g("mouthPucker");
 const raw={
  happy:smile*.72+cheek*.28,
  sad:frown*.52+inner*.34+(1-smile)*.14,
  angry:browDown*.48+eyeSquint*.2+press*.2+sneer*.12,
  surprised:jaw*.42+eyeWide*.32+inner*.26,
  fearful:eyeWide*.38+inner*.34+jaw*.28,
  disgusted:sneer*.55+pucker*.25+frown*.2
 };
 const sum=Object.values(raw).reduce((a,v)=>a+v,0);
 const top=Math.max(...Object.values(raw));
 const neutral=clamp(1-top*.95);
 const all={...raw,neutral:neutral*.55};
 const total=Object.values(all).reduce((a,v)=>a+v,0)||1;
 return labels.map(x=>({...x,value:Math.round(all[x.key]/total*100)})).sort((a,b)=>b.value-a.value);
}
function renderScores(rows){
 $("scores").innerHTML="";
 rows.forEach(x=>{const row=document.createElement("div");row.className="score-row";const emoji=document.createElement("span");emoji.className="score-emoji";emoji.textContent=x.emoji;const mid=document.createElement("div");const meta=document.createElement("div");meta.className="score-meta";const name=document.createElement("span");name.textContent=x.name;const val=document.createElement("span");val.textContent=x.value+"%";meta.append(name,val);const bar=document.createElement("div");bar.className="bar";const fill=document.createElement("i");fill.style.width=x.value+"%";bar.append(fill);mid.append(meta,bar);row.append(emoji,mid);$("scores").append(row);});
}
function resizeCanvas(w,h){canvas.width=w;canvas.height=h;}
function drawLandmarks(result,w,h,flip=false){
 resizeCanvas(w,h);ctx.clearRect(0,0,w,h);
 if(!result.faceLandmarks?.length)return;
 result.faceLandmarks.forEach(points=>{
  if(drawingUtils){ctx.save();if(flip){ctx.translate(w,0);ctx.scale(-1,1)}drawingUtils.drawConnectors(points,FaceLandmarker.FACE_LANDMARKS_TESSELATION,{color:"#66e7d3",lineWidth:.45});drawingUtils.drawConnectors(points,FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,{color:"#a9b7ff",lineWidth:1.4});ctx.restore();}
 });
}
function updateReading(result,mode){
 const count=result.faceLandmarks?.length||0;faceTotal=count;$("facesMetric").textContent=count;$("faceCount").textContent=count+" face"+(count===1?"":"s");$("faceBadge").textContent=count?count+" FACE"+(count===1?"":"S"):"NO FACE";
 if(!count){$("resultEmoji").textContent="—";$("resultName").textContent="No face detected";$("resultDescription").textContent="Try better lighting and face the camera."; $("scores").innerHTML='<div class="score-empty">No face found in this frame.</div>';lastReading=null;return;}
 const rows=scoreExpressions(result),top=rows[0];renderScores(rows);$("resultEmoji").textContent=top.emoji;$("resultName").textContent=top.name;$("resultDescription").textContent=top.desc+" · highest estimated cue level";$("scanCount").textContent=++scans;
 $("modeLabel").textContent=mode.toUpperCase();lastReading={time:new Date(),mode,count,top,rows};addHistory(lastReading);
}
function addHistory(item){
 const stamp=item.time.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
 const key=stamp+"-"+item.top.key;if(history.length&&history[0].key===key)return;
 const entry={...item,key,stamp};history.unshift(entry);history=history.slice(0,12);renderHistory();
}
function renderHistory(){
 const root=$("history");root.innerHTML="";if(!history.length){root.innerHTML='<div class="history-empty">Your session readings will appear here.</div>';return}
 history.forEach(x=>{const row=document.createElement("div");row.className="history-item";const em=document.createElement("span");em.className="hist-emoji";em.textContent=x.top.emoji;const info=document.createElement("div"),b=document.createElement("b"),small=document.createElement("small");b.textContent=x.top.name+" · "+x.top.value+"%";small.textContent=x.mode+" · "+x.count+" face"+(x.count===1?"":"s");info.append(b,small);const time=document.createElement("span");time.className="hist-time";time.textContent=x.stamp;row.append(em,info,time);root.append(row);});
}
function resetResult(){$("resultEmoji").textContent="—";$("resultName").textContent="No analysis yet";$("resultDescription").textContent="Start the camera or upload an image.";$("scores").innerHTML='<div class="score-empty">Analysis results will appear here.</div>';$("faceCount").textContent="0 faces";$("facesMetric").textContent="0";$("faceBadge").textContent="NO FACE";$("modeLabel").textContent="WAITING";lastReading=null;resizeCanvas(1,1);$("scanCount").textContent=scans;}
function stopCamera(){
 if(raf)cancelAnimationFrame(raf);raf=0;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.pause();video.srcObject=null;video.style.display="none";$("stopCamera").disabled=true;$("startCamera").disabled=false;$("switchCamera").disabled=true;if(scanMode==="camera")scanMode="idle";if(!photo.src){$("stageEmpty").hidden=false;setStatus("Standby","Waiting");}else setStatus("Photo loaded","Photo");
}
async function startCamera(){
 try{
  await loadModel();if(stream)stopCamera();if(currentImageUrl){URL.revokeObjectURL(currentImageUrl);currentImageUrl=null;}photo.removeAttribute("src");photo.style.display="none";
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:facing,width:{ideal:1280},height:{ideal:720}},audio:false});
  video.srcObject=stream;video.style.display="block";$("stageEmpty").hidden=true;await video.play();scanMode="camera";$("startCamera").disabled=true;$("stopCamera").disabled=false;$("switchCamera").disabled=false;lastVideoTime=-1;frames=0;lastFpsTime=performance.now();setStatus("Live scanning","Live");
  const loop=()=>{if(!stream||scanMode!=="camera")return;if(video.readyState>=2&&video.currentTime!==lastVideoTime){lastVideoTime=video.currentTime;try{const result=landmarker.detectForVideo(video,performance.now());drawLandmarks(result,video.videoWidth,video.videoHeight,mirror);updateReading(result,"LIVE");frames++;const now=performance.now();if(now-lastFpsTime>=1000){fps=Math.round(frames*1000/(now-lastFpsTime));$("fps").textContent=fps;frames=0;lastFpsTime=now;}}catch(e){console.error(e);}}raf=requestAnimationFrame(loop);};loop();
 }catch(e){console.error(e);alert(e.message||"Camera unavailable. Allow camera permission and use HTTPS.");stopCamera();}
}
async function analysePhoto(file){
 if(!file||!file.type.startsWith("image/"))return;if(file.size>12*1024*1024){alert("Please choose an image smaller than 12 MB.");return;}
 stopCamera();try{await loadModel();if(currentImageUrl)URL.revokeObjectURL(currentImageUrl);currentImageUrl=URL.createObjectURL(file);photo.onload=()=>{video.style.display="none";photo.style.display="block";$("stageEmpty").hidden=true;const w=photo.naturalWidth,h=photo.naturalHeight;const result=landmarker.detect(photo);drawLandmarks(result,w,h,false);scanMode="photo";setStatus("Photo analysed","Photo");$("captureBtn").disabled=false;updateReading(result,"PHOTO");};photo.src=currentImageUrl;}catch(e){console.error(e);alert(e.message||"Could not analyse this image.");}
}
function capture(){
 if(scanMode!=="camera"||!video.videoWidth)return;const out=document.createElement("canvas");out.width=video.videoWidth;out.height=video.videoHeight;const c=out.getContext("2d");if(mirror){c.translate(out.width,0);c.scale(-1,1)}c.drawImage(video,0,0);lastSnapshot=out.toDataURL("image/png");$("downloadCapture").disabled=false;const a=document.createElement("a");a.href=lastSnapshot;a.download="face-scope-capture.png";a.click();
}
function downloadReport(){
 const report={app:"Face Scope - AI Expression Reader",createdAt:new Date().toISOString(),note:"Expression cue estimates only; not a reliable measure of emotion or mental state. Images are not included.",readings:history.map(x=>({time:x.time.toISOString(),mode:x.mode,faces:x.count,leadingCue:x.top.name,estimatedCueLevel:x.top.value,estimatedCueLevels:Object.fromEntries(x.rows.map(r=>[r.name,r.value]))}))};
 const blob=new Blob([JSON.stringify(report,null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="face-scope-session-report.json";a.click();URL.revokeObjectURL(url);
}
$("loginForm").addEventListener("submit",async e=>{e.preventDefault();$("loginButton").disabled=true;setMessage("loginMsg","Signing in…");try{const r=await auth.signInWithEmailAndPassword($("email").value.trim(),$("password").value);if(r.user.uid!==ADMIN_UID){await auth.signOut();setMessage("loginMsg","This account is not authorized.",true);}}catch(e){console.error(e);setMessage("loginMsg","Sign-in failed. Check your email and password.",true);}finally{$("loginButton").disabled=false;}});
$("logout").onclick=async()=>{stopCamera();await auth.signOut();};
auth.onAuthStateChanged(async u=>{if(!u){user=null;$("app").hidden=true;$("login").hidden=false;stopCamera();return;}if(u.uid!==ADMIN_UID){await auth.signOut();setMessage("loginMsg","This account is not authorized.",true);return;}user=u;$("login").hidden=true;$("app").hidden=false;});
$("startCamera").onclick=startCamera;$("stopCamera").onclick=stopCamera;
$("switchCamera").onclick=async()=>{facing=facing==="user"?"environment":"user";await startCamera();};
$("photoInput").onchange=e=>{const file=e.target.files?.[0];if(file)analysePhoto(file);e.target.value="";};
$("mirrorToggle").onclick=()=>{mirror=!mirror;video.style.transform=mirror?"scaleX(-1)":"none";$("mirrorToggle").textContent="◐ Mirror: "+(mirror?"on":"off");};
$("captureBtn").onclick=capture;$("downloadCapture").onclick=()=>{if(!lastSnapshot)return;const a=document.createElement("a");a.href=lastSnapshot;a.download="face-scope-capture.png";a.click();};
$("downloadReport").onclick=downloadReport;$("clearHistory").onclick=()=>{history=[];renderHistory();};$("clearBtn").onclick=()=>{stopCamera();if(currentImageUrl)URL.revokeObjectURL(currentImageUrl);currentImageUrl=null;photo.removeAttribute("src");photo.style.display="none";$("stageEmpty").hidden=false;$("captureBtn").disabled=true;$("downloadCapture").disabled=true;lastSnapshot=null;resetResult();setStatus("Standby","Waiting");};
$("themeToggle").onclick=()=>document.body.classList.toggle("light");
window.addEventListener("beforeunload",()=>{if(stream)stream.getTracks().forEach(t=>t.stop());if(currentImageUrl)URL.revokeObjectURL(currentImageUrl);});
