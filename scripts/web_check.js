#!/usr/bin/env node
// Browser-free sanity checks for the web front end:
//  * app.js parses;
//  * every element id referenced via $('id') exists in index.html;
//  * no absolute http(s):// asset references (the LAN may block CDNs);
//  * every /api path the front end calls exists as a route in server.py.
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const root = path.resolve(__dirname, '..');
const st = path.join(root, 'ai_movie', 'web', 'static');
const js = fs.readFileSync(path.join(st, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(st, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(st, 'app.css'), 'utf8');
const server = fs.readFileSync(path.join(root, 'ai_movie', 'web', 'server.py'), 'utf8');
let bad = 0;
const fail = (m) => { console.error('FAIL ' + m); bad++; };

execFileSync('node', ['--check', path.join(st, 'app.js')]);
console.log('ok   app.js parses');

const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
for (const m of js.matchAll(/\$\('([a-z0-9-]+)'\)/g)) {
  if (!ids.has(m[1]) && !/^opt-/.test(m[1])) fail(`missing element id #${m[1]}`);
}
console.log(`ok   ${ids.size} ids in index.html`);

for (const [name, text] of [['app.js', js], ['index.html', html], ['app.css', css]]) {
  if (/https?:\/\//.test(text.replace(/\/\/.*$/gm, ''))) fail(`${name} references an absolute http(s) URL`);
}
console.log('ok   no external asset URLs');

const routes = [...server.matchAll(/@app\.(get|post|put|delete|api_route)\("([^"]+)"/g)].map((m) => m[2]);
const norm = (p) => p.replace(/\$\{[^}]+\}/g, 'X').replace(/\{[^}]+\}/g, 'X').replace(/\?.*$/, '');
const routeSet = new Set(routes.map(norm));
const jsFlat = js.replace(/\$\{[^}]*\}/g, 'X');
for (const m of jsFlat.matchAll(/(\/api\/[A-Za-z0-9_\/X.-]+)/g)) {
  const p = norm(m[1]);
  if (!routeSet.has(p)) fail(`front end calls ${m[1]} but server has no such route`);
}
console.log(`ok   ${routes.length} routes checked`);
process.exit(bad ? 1 : 0);
