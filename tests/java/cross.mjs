import { FireScan } from '../../web/js/firescan.js';
const W = 160, H = 120;
const paint = (p, f) => { const d = new Uint8ClampedArray(W*H*4);
  for (let y=0;y<H;y++) for (let x=0;x<W;x++){ const [r,g,b]=p(x,y,f); const i=(y*W+x)*4;
    d[i]=r;d[i+1]=g;d[i+2]=b;d[i+3]=255;} return d; };
const ground=(x,y)=>{const n=((x*7+y*13)%11)*6;return [28+n,58+n,22+(n>>1)];};
const inside=(x,y,b)=>x>=b[0]&&x<b[2]&&y>=b[1]&&y<b[3];
const FIRE=[40,60,80,100];
function run(name, frames, p){ const s=new FireScan(); let out=[];
  for(let f=0;f<frames;f++) out=s.scanPixels(paint(p,f),W,H);
  const said=out.map(r=>`${r.label} ${r.confidence.toFixed(2)} [${r.box.map(v=>v.toFixed(2)).join(' ')}]`).sort();
  console.log(`${name}: ${said.length?said.join(' | '):'-'}`); }
run('flame',20,(x,y,f)=>{ if(!inside(x,y,FIRE)) return ground(x,y);
  const h=22+14*Math.sin(f*1.1+x*0.35); if(y<FIRE[3]-h) return ground(x,y);
  return ((x*3+y*5+f*7)%9)===0?[120,45,12]:[255,140,30];});
run('static-panel',20,(x,y)=>inside(x,y,FIRE)?[255,140,30]:ground(x,y));
run('moving-van',20,(x,y,f)=>{const b=[10+f*4,60,50+f*4,90];return inside(x,y,b)?[255,140,30]:ground(x,y);});
run('sunset',20,(x,y,f)=>{ if(y>=55) return ground(x,y); const d=f*0.4, n=((x+y*3+f)%5)-2;
  return [252-d+n,138-y+n,40+y+n];});
run('smoke',24,(x,y,f)=>{ const a=Math.max(0,f-5), top=Math.max(0,100-a*6);
  const inP=f>5&&y>=top&&y<100&&Math.abs(x-80)<30+((y+f)%7); if(!inP) return ground(x,y);
  const g=168+((x+y*2+f*9)%5); return [g,g+2,g-1];});
run('overcast',24,(x,y,f)=>{const w=(f%3)-1; if(y<60+w){const g=176+((x+f)%4);return [g,g,g+1];} return ground(x,y);});
run('nothing',20,(x,y,f)=>ground(x+f,y));
run('office',24,(x,y,f)=>{ const shift=f*3;
  if(Math.abs(x-(30+shift))<26&&y>30) return [46,40,38];
  const w=190+((x+y)%2); return [w,w+1,w-1];});
import { tileRegion, overlap } from '../../web/js/tiles.js';
{
  const parts=[]; for(let i=0;i<6;i++){const r=tileRegion(i,1920,1080);
    parts.push(`[${r.x.toFixed(1)} ${r.y.toFixed(1)} ${r.width.toFixed(1)} ${r.height.toFixed(1)}]`);}
  console.log('tiles: '+parts.join(' '));
  const a=[100,100,140,190], b=[104,98,144,188], far=[400,100,440,190];
  console.log(`overlap: ${overlap(a,b).toFixed(4)} ${overlap(a,far).toFixed(4)}`);
}
run('still',1,(x,y)=>{ if(!inside(x,y,FIRE)) return ground(x,y);
  return ((x*3+y*5)%10)>2?[255,140,30]:[120,40,10];});

// The YOLO decode, which exists twice for the same reason the flame scanner does. This
// half is the reference; Yolo.java is compared against it below.
{
  const { decodeHead, iou, unletterbox } = await import('../../web/js/yolo.js');

  // A head made of arithmetic rather than of random numbers, so both languages can build
  // the identical array without one having to ship the other a file.
  const CHANNELS = 15, ANCHORS = 2100;
  const value = (i) => Math.fround((((i * 1103515245 + 12345) >>> 8) & 0xffff) / 65535);
  const head = new Float32Array(CHANNELS * ANCHORS);
  for (let i = 0; i < head.length; i += 1) head[i] = value(i);

  const say = (d) => `${d.classId}:${d.confidence.toFixed(6)}`
    + `[${d.box.map((v) => v.toFixed(6)).join(' ')}]`;

  const all = decodeHead(head, CHANNELS, ANCHORS, { confThreshold: 0.25, iouThreshold: 0.45 });
  console.log(`yolo-all: ${all.length} ${all.slice(0, 6).map(say).join(' | ')}`);

  const people = decodeHead(head, CHANNELS, ANCHORS,
    { confThreshold: 0.25, iouThreshold: 0.45, keepClasses: [0, 1] });
  console.log(`yolo-people: ${people.length} ${people.slice(0, 6).map(say).join(' | ')}`);

  // The transposed layout. The TFLite converter emits [1, anchors, 4+nc] where ONNX gives
  // [1, 4+nc, anchors], so the Java rearranges before decoding; this is the same
  // rearrangement written the obvious way, which is what it has to agree with.
  const perAnchor = new Float32Array(CHANNELS * ANCHORS);
  for (let i = 0; i < perAnchor.length; i += 1) perAnchor[i] = value(i * 7 + 3);
  const majored = new Float32Array(CHANNELS * ANCHORS);
  for (let a = 0; a < ANCHORS; a += 1) {
    for (let c = 0; c < CHANNELS; c += 1) majored[c * ANCHORS + a] = perAnchor[a * CHANNELS + c];
  }
  const transposed = decodeHead(majored, CHANNELS, ANCHORS,
    { confThreshold: 0.25, iouThreshold: 0.45, keepClasses: [0, 1] });
  console.log(`yolo-transposed: ${transposed.length} ${transposed.slice(0, 6).map(say).join(' | ')}`);

  const a = [100, 100, 140, 190], b = [104, 98, 144, 188], far = [400, 100, 440, 190];
  console.log(`yolo-iou: ${iou(a, b).toFixed(6)} ${iou(a, far).toFixed(6)} ${iou(a, a).toFixed(6)}`);
  console.log('yolo-unletterbox: '
    + unletterbox([12, 30, 300, 290], 0.3333, 0, 46.5, 1920, 1080)
      .map((v) => v.toFixed(6)).join(' '));
}

// The tracker under a camera that is moving. See Cross.tracking().
{
  const { Tracker } = await import('../../web/js/track.js');
  for (const speed of [0, 20, 45]) {
    const tracker = new Tracker();
    let clock = 1000;
    for (let step = 0; step < 8; step += 1) {
      const crowd = [];
      for (let i = 0; i < 30; i += 1) {
        const x = (i % 6) * 90 - step * speed;
        const y = Math.floor(i / 6) * 120;
        crowd.push({ label: 'person', confidence: 0.8, box: [x, y, x + 40, y + 80] });
      }
      tracker.update(crowd, clock);
      clock += 250;
    }
    console.log(`track-pan-${speed}: ${tracker.countSeen('person')}`);
  }

  // And the sparse scene, at the size a person actually is from altitude. See Cross.
  for (const people of [1, 2, 3]) {
    const counts = [];
    for (const speed of [0, 20, 45, 60, 90]) {
      const tracker = new Tracker();
      let clock = 1000;
      for (let step = 0; step < 12; step += 1) {
        const few = [];
        for (let i = 0; i < people; i += 1) {
          const x = i * 120 - step * speed;
          few.push({ label: 'person', confidence: 0.8, box: [x, 100, x + 10, 122] });
        }
        tracker.update(few, clock);
        clock += 250;
      }
      counts.push(tracker.countSeen('person'));
    }
    console.log(`track-sparse-${people}: ${counts.join(' ')}`);
  }
}
