import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';
import http from 'http';

const POSES = JSON.parse(fs.readFileSync('pipeline/render_views/camera_poses.json'));
const SPLAT = 'http://localhost:8081/3dgs_compressed.ply';
const OUT_DIR = 'pipeline/output/views';
fs.mkdirSync(OUT_DIR, { recursive: true });

const staticServer = http.createServer((req, res) => {
  const filepath = 'data/scene/demo' + req.url;
  if (!fs.existsSync(filepath)) { res.statusCode = 404; res.end(); return; }
  res.end(fs.readFileSync(filepath));
}).listen(8081);

const viewerServer = http.createServer((req, res) => {
  res.setHeader('Content-Type', 'text/html');
  res.end(fs.readFileSync('pipeline/render_views/index.html'));
}).listen(8082);

const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1024, height: 768 });

for (let i = 0; i < POSES.length; i++) {
  const p = POSES[i];
  const url = `http://localhost:8082/?splat=${encodeURIComponent(SPLAT)}&x=${p.position[0]}&y=${p.position[1]}&z=${p.position[2]}&tx=${p.lookAt[0]}&ty=${p.lookAt[1]}&tz=${p.lookAt[2]}`;
  await page.goto(url);
  try {
    await page.waitForFunction(() => window._ready === true, { timeout: 45000 });
  } catch (e) {
    console.warn(`view ${i}: readiness timeout, screenshotting anyway`);
  }
  await new Promise(r => setTimeout(r, 1500));
  const outPath = path.join(OUT_DIR, `view_${String(i).padStart(3, '0')}.png`);
  await page.screenshot({ path: outPath, fullPage: false });
  console.log(`rendered ${i+1}/${POSES.length}`);
}

await browser.close();
staticServer.close();
viewerServer.close();

fs.writeFileSync('pipeline/output/views/_intrinsics.json', JSON.stringify({
  width: 1024, height: 768, fov_vertical_deg: 60.0, poses: POSES,
}, null, 2));
console.log('done');
