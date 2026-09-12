/* AI Movie 配音工作台 — single-page front end (no build step, no external assets). */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined) continue;
      el.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
    }
    return el;
  };
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmtSize = (n) => n == null ? '' : n > 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n > 1e6 ? (n / 1e6).toFixed(1) + ' MB' : Math.round(n / 1e3) + ' KB';
  const fmtT = (t) => { if (t == null) return ''; const m = Math.floor(t / 60), s = (t % 60); return `${m}:${s.toFixed(1).padStart(4, '0')}`; };

  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts,
      body: opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body });
    let data = null;
    try { data = await r.json(); } catch (e) { data = null; }
    if (!r.ok) {
      const msg = (data && (data.detail || data.error)) || r.statusText;
      const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      err.status = r.status; err.data = data; throw err;
    }
    return data;
  }
  function toast(msg, isErr = false) {
    const t = h('div', { class: 'toast' + (isErr ? ' err' : '') }, msg);
    document.body.appendChild(t);
    setTimeout(() => t.remove(), isErr ? 6000 : 2500);
  }

  const STATE = { project: null, data: null, step: null, job: null, es: null, seq: 0, segments: null, videoSrc: 'final' };
  const STEP_ORDER_LABELS = {};

  // ── projects list ──
  async function loadProjects() {
    const rows = await api('/api/projects');
    const box = $('projects'); box.innerHTML = '';
    for (const p of rows) {
      const dot = h('span', { class: 'dot' + (p.job ? ' running' : '') });
      const el = h('div', { class: 'proj' + (STATE.project === p.name ? ' active' : ''), onclick: () => openProject(p.name) },
        h('span', {}, p.name, h('small', { class: 'muted' }, ` ${p.steps_done}/15${p.has_vc ? ' +v2' : ''}`)), dot);
      box.appendChild(el);
    }
    const hl = await api('/api/health');
    $('gpu-note').textContent = hl.gpu_busy ? '⚠ GPU 被会话外的任务占用（终端里的流水线），新任务会被拒绝或排队' : '';
  }

  // ── project ──
  async function openProject(name) {
    STATE.project = name;
    await refreshProject();
    if (!STATE.step) selectStep(firstInteresting());
    loadProjects();
  }
  async function refreshProject() {
    if (!STATE.project) return;
    STATE.data = await api(`/api/projects/${encodeURIComponent(STATE.project)}`);
    $('title').textContent = STATE.project;
    const s = STATE.data.summary;
    $('subtitle').textContent = `${fmtT(s.duration)} · ${s.n_segments} 段 · 说话人 ${Object.entries(s.speakers).map(([k, v]) => k + (v.gender === 'male' ? '♂' : '♀')).join(' ') || '—'}`
      + (s.qc && s.qc.fit ? ` · QC ${s.qc.fit.PASS}/${s.qc.fit.WARN}/${s.qc.fit.FAIL}` : '');
    renderToolbar();
    if (STATE.data.job && (!STATE.job || STATE.job.id !== STATE.data.job.id)) attachJob(STATE.data.job);
  }
  function firstInteresting() {
    const st = STATE.data.steps;
    for (const k of STATE.data.step_order) if (['ready', 'stale', 'failed', 'running'].includes(st[k].status)) return k;
    return 'compose';
  }
  function renderToolbar() {
    const tb = $('toolbar'); tb.innerHTML = '';
    for (const k of STATE.data.step_order) {
      const st = STATE.data.steps[k];
      STEP_ORDER_LABELS[k] = st.label;
      const b = h('div', { class: `step ${st.status}${STATE.step === k ? ' selected' : ''}`, title: (st.reasons || []).join('\n'), onclick: () => selectStep(k) },
        st.label, h('small', {}, statusLabel(st.status)));
      tb.appendChild(b);
    }
  }
  const statusLabel = (s) => ({ locked: '未就绪', ready: '可运行', done: '完成', stale: '需重跑', running: '运行中', failed: '失败', skipped: '已跳过', unknown: '?' }[s] || s);

  function selectStep(k) {
    STATE.step = k; renderToolbar(); renderPane();
  }

  // ── panes ──
  async function renderPane() {
    const main = $('main'); main.innerHTML = '';
    const k = STATE.step, d = STATE.data, st = d.steps[k];
    const head = h('div', { class: 'card' },
      h('div', { class: 'row', style: 'justify-content:space-between' },
        h('div', {}, h('h3', {}, `${st.label} `, h('span', { class: 'badge ' + (st.status === 'done' ? 'ok' : st.status === 'stale' ? 'warn' : st.status === 'failed' ? 'bad' : '') }, statusLabel(st.status))),
          st.reasons && st.reasons.length ? h('div', { class: 'small muted' }, st.reasons.join('；')) : null),
        actionButtons(k)));
    main.appendChild(head);
    const opts = optionsPanel(k);
    if (opts) main.appendChild(opts);
    try {
      const pane = await stepPane(k);
      if (pane) main.appendChild(pane);
    } catch (e) { main.appendChild(h('div', { class: 'card' }, '加载失败：' + e.message)); }
  }
  function actionButtons(k) {
    const busy = !!STATE.data.job;
    const wrap = h('div', { class: 'row' });
    if (k === 'v2') {
      wrap.appendChild(h('button', { class: 'primary', disabled: busy || null, onclick: () => startJob({ kind: 'v2' }) }, '生成 v2（原声音色）'));
    } else if (k === 'deliver') {
      wrap.appendChild(h('button', { class: 'primary', disabled: busy || null, onclick: () => startJob({ kind: 'deliver', version: 'vc' }) }, '生成交付包(v2)'));
      wrap.appendChild(h('button', { disabled: busy || null, onclick: () => startJob({ kind: 'deliver', version: 'v1' }) }, '交付包(v1)'));
    } else {
      wrap.appendChild(h('button', { class: 'primary', disabled: busy || null, onclick: () => startJob({ kind: 'steps', steps: [k] }) }, '运行此步'));
      wrap.appendChild(h('button', { disabled: busy || null, onclick: () => startJob({ kind: 'steps', steps: [k], force: true }) }, '强制重跑'));
      wrap.appendChild(h('button', { disabled: busy || null, title: '从此步到合成视频，跳过仍有效的步骤', onclick: () => startJob({ kind: 'steps', steps: fromHere(k) }) }, '从此步往后'));
    }
    return wrap;
  }
  function fromHere(k) {
    const order = STATE.data.step_order.filter((s) => !['v2', 'deliver'].includes(s));
    return order.slice(order.indexOf(k));
  }

  // options per step
  const OPTS = {
    asr: [['language', '语言', 'text'], ['asr_backend', 'ASR 后端', 'select', ['openai-whisper', 'faster-whisper']], ['num_speakers', '说话人数(空=自动)', 'number'], ['no_diarize', '不做说话人日志', 'bool']],
    osd: [['no_osd', '跳过重叠语音检测', 'bool']],
    glossary: [['translate_helper', '术语抽取模型(可空)', 'text']],
    translate: [['engines', '翻译引擎(逗号分隔)', 'select-multi'], ['chosen_engine', '采用引擎(可空)', 'text']],
    tts: [['voice_mode', '声音模式', 'select', ['sft', 'clone', 'gender', 'female', 'male']], ['no_ref_probe', '跳过参考音探针', 'bool']],
    compact: [['no_compact', '跳过台词压缩', 'bool']],
    lipsync: [['lipsync_backend', '口型引擎', 'select', ['musetalk', 'wav2lip']], ['lipsync_audio_offset_ms', '音频偏移 ms', 'number'], ['fusion', '融合', 'select', ['', 'alpha', 'laplacian']], ['occlusion_mode', '遮挡模式', 'select', ['', 'frame', 'region']]],
    enhance: [['enhance_fidelity', 'CodeFormer 保真度 0–1', 'number'], ['enhance_protect_lips', '保护嘴唇(0/1)', 'number']],
  };
  function optionsPanel(k) {
    const spec = OPTS[k]; if (!spec) return null;
    const o = STATE.data.options;
    const form = h('div', { class: 'row' });
    for (const [key, label, type, choices] of spec) {
      let input;
      if (type === 'bool') input = h('input', { type: 'checkbox', id: 'opt-' + key });
      else if (type === 'select') { input = h('select', { id: 'opt-' + key }, ...choices.map((c) => h('option', { value: c }, c || '(默认)'))); }
      else if (type === 'select-multi') {
        input = h('select', { id: 'opt-' + key }, ...STATE.data.engines.map((c) => h('option', { value: c }, c)));
      } else input = h('input', { type: type === 'number' ? 'text' : 'text', id: 'opt-' + key, style: 'width:120px' });
      if (type === 'bool') input.checked = !!o[key]; else input.value = o[key] == null ? '' : o[key];
      form.appendChild(h('label', { class: 'chk' }, label, input));
    }
    form.appendChild(h('button', { onclick: async () => {
      const body = {};
      for (const [key, , type] of spec) {
        const el = $('opt-' + key);
        body[key] = type === 'bool' ? el.checked : (el.value === '' ? null : el.value);
      }
      try { await api(`/api/projects/${STATE.project}/options`, { method: 'PUT', body }); toast('选项已保存'); await refreshProject(); renderPane(); }
      catch (e) { toast(e.message, true); }
    } }, '保存选项'));
    return h('div', { class: 'card' }, h('h3', {}, '选项'), form, h('div', { class: 'small muted' }, '改选项后相关步骤会显示“需重跑”。'));
  }

  async function stepPane(k) {
    switch (k) {
      case 'demux': case 'separate': return audioPane(k);
      case 'osd': return osdPane();
      case 'asr': return asrPane();
      case 'glossary': return glossaryPane();
      case 'translate': return translatePane();
      case 'tts': case 'compact': case 'fit': return ttsPane(k);
      case 'mix': return mixPane();
      case 'faces': return facesPane();
      case 'lipsync': case 'enhance': case 'compose': case 'v2': return videoPane(k);
      case 'qc': return qcPane();
      case 'deliver': return deliverPane();
      default: return null;
    }
  }

  async function audioPane(k) {
    const v = STATE.data.videos;
    const card = h('div', { class: 'card' }, h('h3', {}, '产物'));
    if (k === 'demux') { card.appendChild(h('div', {}, '原片：', v.original ? h('video', { controls: '', src: v.original, preload: 'metadata' }) : '—')); }
    else {
      const segs = await getSegments();
      card.appendChild(h('div', { class: 'small muted' }, '人声分离产物（16k 分析轨 / 全采样率床）在 workspace/<name>/separated/。'));
    }
    return card;
  }
  async function osdPane() {
    const s = STATE.data.summary.osd || {};
    return h('div', { class: 'card' }, h('h3', {}, '重叠语音'), h('div', {}, s.available ? `检测到重叠共 ${s.total_overlap_s} s（pyannote/segmentation-3.0，CPU）` : `不可用：${s.reason || '未运行'}`));
  }

  async function getSegments(force = false) {
    if (!STATE.segments || force) STATE.segments = await api(`/api/projects/${STATE.project}/segments`);
    return STATE.segments;
  }

  async function asrPane() {
    const [segs, spk] = await Promise.all([getSegments(true), api(`/api/projects/${STATE.project}/speakers`)]);
    const spkIds = Object.keys(spk.speakers);
    const card = h('div', { class: 'card' }, h('h3', {}, `转写与说话人（${segs.length} 段）`));
    const sp = h('div', { class: 'row' });
    for (const [id, m] of Object.entries(spk.speakers)) {
      sp.appendChild(h('span', { class: 'badge' }, `${id} ${m.gender === 'male' ? '♂' : '♀'} ${m.n_segments}段 ${m.f0_median ? m.f0_median + 'Hz' : ''}`),
        m.demo_url ? h('audio', { controls: '', src: m.demo_url, preload: 'none' }) : null);
    }
    sp.appendChild(h('button', { onclick: async () => {
      const g = prompt('新建说话人性别（male/female）', 'male'); if (!g) return;
      try { await api(`/api/projects/${STATE.project}/speakers`, { method: 'POST', body: { gender: g } }); toast('已新建'); renderPane(); } catch (e) { toast(e.message, true); }
    } }, '新增说话人'));
    sp.appendChild(h('button', { disabled: STATE.data.job ? '' : null, onclick: () => startJob({ kind: 'review_speakers' }) }, '生成说话人复核清单'));
    card.appendChild(sp);
    const tbl = h('table', {}, h('thead', {}, h('tr', {}, ...['#', '时间', '说话人', '性别', '置信', '重叠', '日文', ''].map((t) => h('th', {}, t)))));
    const tb = h('tbody');
    for (const s of segs) {
      const sel = h('select', {}, ...spkIds.map((id) => h('option', { value: id, selected: id === s.speaker ? '' : null }, id)));
      const gsel = h('select', {}, ...['female', 'male'].map((g) => h('option', { value: g, selected: g === s.gender ? '' : null }, g === 'male' ? '男' : '女')));
      const save = h('button', { class: 'small', onclick: async () => {
        try {
          const body = sel.value !== s.speaker ? { speaker: sel.value } : { gender: gsel.value };
          await api(`/api/projects/${STATE.project}/segments/${s.idx}/speaker`, { method: 'PUT', body });
          toast(`第 ${s.idx} 段已更新`); STATE.segments = null; await refreshProject(); renderPane();
        } catch (e) { toast(e.message, true); }
      } }, '保存');
      tb.appendChild(h('tr', {}, h('td', {}, String(s.idx)), h('td', { class: 'mono small' }, s.tc), h('td', {}, sel), h('td', {}, gsel),
        h('td', { class: 'small' }, `${s.speaker_conf ?? ''}/${s.asr_conf ?? ''}`), h('td', { class: 'small' }, s.overlap ? h('span', { class: 'badge ' + (s.overlap > 0.6 ? 'bad' : s.overlap > 0.3 ? 'warn' : '') }, Math.round(s.overlap * 100) + '%') : ''),
        h('td', {}, s.text || ''), h('td', {}, save)));
    }
    tbl.appendChild(tb);
    card.appendChild(h('div', { style: 'max-height:60vh;overflow:auto' }, tbl));
    const rev = await api(`/api/projects/${STATE.project}/review`);
    if (rev.available && rev.items.length) {
      const rc = h('div', { class: 'card' }, h('h3', {}, `说话人复核清单（${rev.items.length} 条建议改动）`));
      for (const it of rev.items) rc.appendChild(h('div', { class: 'row' }, h('span', { class: 'mono small' }, `#${it.index} ${it.current}→${it.proposed} (${it.decided_by})`), it.clip_url ? h('audio', { controls: '', src: it.clip_url, preload: 'none' }) : null, h('span', { class: 'small' }, it.text || '')));
      card.appendChild(rc);
    }
    return card;
  }

  async function glossaryPane() {
    const g = await api(`/api/projects/${STATE.project}/glossary`);
    const card = h('div', { class: 'card' }, h('h3', {}, '术语表'), h('div', { class: 'small muted' }, '日文 → 中文译法。保存后翻译步会“需重跑”；“仅应用到现有译文”只做字符串替换，不重译。'));
    const tb = h('tbody');
    const addRow = (ja = '', zh = '', kind = 'name') => tb.appendChild(h('tr', {}, h('td', {}, h('input', { value: ja, class: 'ja' })), h('td', {}, h('input', { value: zh, class: 'zh' })), h('td', {}, h('input', { value: kind, class: 'kind', style: 'width:70px' })), h('td', {}, h('button', { onclick: (e) => e.target.closest('tr').remove() }, '删'))));
    for (const [ja, v] of Object.entries(g)) addRow(ja, v.zh || v, v.kind || 'name');
    const collect = () => { const t = {}; for (const tr of tb.querySelectorAll('tr')) { const ja = tr.querySelector('.ja').value.trim(); if (ja) t[ja] = { zh: tr.querySelector('.zh').value.trim(), kind: tr.querySelector('.kind').value.trim() || 'name', source: 'user' }; } return t; };
    const saveBtn = (apply) => h('button', { class: apply ? '' : 'primary', onclick: async () => {
      try { const r = await api(`/api/projects/${STATE.project}/glossary`, { method: 'PUT', body: { terms: collect(), apply_to_translation: apply } }); toast(apply ? `已应用到 ${r.applied} 句` : '术语表已保存'); await refreshProject(); renderPane(); } catch (e) { toast(e.message, true); }
    } }, apply ? '仅应用到现有译文' : '保存术语表');
    card.appendChild(h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, '日文'), h('th', {}, '中文'), h('th', {}, '类型'), h('th'))), tb));
    card.appendChild(h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { onclick: () => addRow() }, '+ 添加'), saveBtn(false), saveBtn(true)));
    return card;
  }

  async function translatePane() {
    const segs = await getSegments(true);
    const card = h('div', { class: 'card' }, h('h3', {}, '译文（点击译文单元格编辑，回车保存）'));
    const tbl = h('table', {}, h('thead', {}, h('tr', {}, ...['#', '时间', '说话人', '日文', '中文译文', '压缩前原译'].map((t) => h('th', {}, t)))));
    const tb = h('tbody');
    for (const s of segs) {
      const cell = h('td', { class: 'edit', title: '点击编辑' }, s.text_translated || '');
      cell.addEventListener('click', () => { if (cell.getAttribute('contenteditable') === 'true') return; cell.setAttribute('contenteditable', 'true'); cell.focus(); });
      const commit = async () => {
        cell.removeAttribute('contenteditable');
        const t = cell.textContent.trim();
        if (t === (s.text_translated || '')) return;
        try { await api(`/api/projects/${STATE.project}/segments/${s.idx}/translation`, { method: 'PUT', body: { text: t } }); s.text_translated = t; toast(`第 ${s.idx} 段译文已保存`); await refreshProject(); }
        catch (e) { toast(e.message, true); cell.textContent = s.text_translated || ''; }
      };
      cell.addEventListener('blur', commit);
      cell.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); cell.blur(); } if (e.key === 'Escape') { cell.textContent = s.text_translated || ''; cell.blur(); } });
      tb.appendChild(h('tr', {}, h('td', {}, String(s.idx)), h('td', { class: 'mono small' }, s.tc), h('td', {}, `${s.speaker || ''}${s.gender === 'male' ? '♂' : '♀'}`), h('td', {}, s.text || ''), cell, h('td', { class: 'small muted' }, s.text_translated_full || '')));
    }
    tbl.appendChild(tb);
    card.appendChild(h('div', { style: 'max-height:65vh;overflow:auto' }, tbl));
    return card;
  }

  async function ttsPane(k) {
    const segs = await getSegments(true);
    const card = h('div', { class: 'card' }, h('h3', {}, k === 'compact' ? '台词压缩与合成音频' : '合成音频（自然 / 拟合时间槽 / v2 原声）'));
    if (k === 'compact') {
      const c = await api(`/api/projects/${STATE.project}/compact`);
      card.appendChild(h('div', { class: 'small muted' }, c.skipped ? '本工程跳过了压缩。' : `尝试 ${c.attempted ?? 0} 句，接受 ${c.rewritten ?? 0} 句。`));
      if (c.report && c.report.length) {
        const t = h('table', {}, h('thead', {}, h('tr', {}, ...['#', '状态', '槽(s)', '比例 前→后', '原译', '压缩后'].map((x) => h('th', {}, x)))));
        for (const r of c.report) t.appendChild(h('tr', {}, h('td', {}, String(r.idx)), h('td', {}, r.status), h('td', {}, String(r.slot)), h('td', {}, `${r.ratio_before} → ${r.ratio_after}`), h('td', {}, r.text_full || ''), h('td', {}, r.text_compact || '')));
        card.appendChild(t);
      }
    }
    const tbl = h('table', {}, h('thead', {}, h('tr', {}, ...['#', '时间', '说话人', '译文', '自然', '拟合', 'v2', '指标'].map((t) => h('th', {}, t)))));
    const tb = h('tbody');
    for (const s of segs) {
      const badges = [];
      if (s.fit_ratio && s.fit_ratio > 1.25) badges.push(h('span', { class: 'badge ' + (s.fit_ratio > 1.6 ? 'bad' : 'warn') }, `×${s.fit_ratio}`));
      if (s.overrun) badges.push(h('span', { class: 'badge warn' }, `截${s.overrun}s`));
      if (s.tts_fallback) badges.push(h('span', { class: 'badge' }, '内置音色'));
      if (s.compact === 'rewritten') badges.push(h('span', { class: 'badge' }, '已压缩'));
      if (s.tts_error) badges.push(h('span', { class: 'badge bad' }, '合成失败'));
      if (s.mix_gain_db != null) badges.push(h('span', { class: 'badge' }, `${s.mix_gain_db > 0 ? '+' : ''}${s.mix_gain_db} dB`));
      tb.appendChild(h('tr', {}, h('td', {}, String(s.idx)), h('td', { class: 'mono small' }, s.tc), h('td', {}, s.speaker || ''), h('td', {}, s.text_translated || ''),
        h('td', {}, s.audio_url ? h('audio', { controls: '', src: s.audio_url, preload: 'none' }) : ''), h('td', {}, s.audio_fit_url ? h('audio', { controls: '', src: s.audio_fit_url, preload: 'none' }) : ''),
        h('td', {}, s.vc_audio_url ? h('audio', { controls: '', src: s.vc_audio_url, preload: 'none', title: s.vc ? '克隆' : '内置' }) : ''), h('td', {}, ...badges)));
    }
    tbl.appendChild(tb);
    card.appendChild(h('div', { style: 'max-height:60vh;overflow:auto' }, tbl));
    const spk = await api(`/api/projects/${STATE.project}/speakers`);
    const abs = Object.entries(spk.speakers).filter(([, m]) => m.ab_url);
    if (abs.length) card.appendChild(h('div', { class: 'row' }, '原声 vs 克隆 A/B：', ...abs.map(([id, m]) => h('span', {}, id, h('audio', { controls: '', src: m.ab_url, preload: 'none' })))));
    return card;
  }
  async function mixPane() {
    const v = STATE.data.videos;
    return h('div', { class: 'card' }, h('h3', {}, '混音'), v.mix ? h('div', {}, 'v1 混音：', h('audio', { controls: '', src: v.mix, preload: 'none' })) : '—', v.vc_mix ? h('div', {}, 'v2 混音：', h('audio', { controls: '', src: v.vc_mix, preload: 'none' })) : null);
  }

  async function facesPane() {
    const f = await api(`/api/projects/${STATE.project}/faces`);
    const card = h('div', { class: 'card' }, h('h3', {}, `人脸轨迹（${f.tracks.length} 条，切镜 ${f.cuts} 处，锚定 ${f.anchored_frames}/${f.n_frames || '?'} 帧）`));
    const thumbs = h('div', { class: 'thumbs' });
    for (const t of f.tracks) thumbs.appendChild(h('div', { class: 'thumb' }, t.thumb_url ? h('img', { src: t.thumb_url }) : null, h('div', {}, `轨迹 ${t.id} ${t.gender === 'male' ? '♂' : t.gender === 'female' ? '♀' : '?'} 置信 ${t.conf ?? ''}`), h('div', { class: 'muted' }, `帧 ${t.first}–${t.last}`)));
    card.appendChild(thumbs);
    const bind = h('div', { class: 'card' }, h('h3', {}, '说话人 ↔ 人脸绑定'), h('div', { class: 'small muted' }, '“画外”= 该说话人不动画面。改完保存后需重跑 人物锚定 → 口型 → 增强 → 合成。'));
    const sels = {};
    const cur = f.override ? Object.fromEntries(f.override.split(',').map((p) => p.split('='))) : {};
    for (const [spk, tid] of Object.entries(f.speaker_track)) {
      const sel = h('select', {}, h('option', { value: 'none' }, '画外/不绑定'), ...f.tracks.map((t) => h('option', { value: String(t.id) }, `轨迹 ${t.id} (${t.gender || '?'})`)));
      sel.value = cur[spk] != null ? String(cur[spk]) : (tid == null ? 'none' : String(tid));
      sels[spk] = sel;
      bind.appendChild(h('div', { class: 'row' }, h('strong', {}, spk), sel, h('span', { class: 'small muted' }, `自动结果：${tid == null ? '画外' : '轨迹 ' + tid}`)));
    }
    bind.appendChild(h('button', { class: 'primary', onclick: async () => {
      const body = {}; for (const [s, sel] of Object.entries(sels)) body[s] = sel.value === 'none' ? null : Number(sel.value);
      try { await api(`/api/projects/${STATE.project}/faces/binding`, { method: 'PUT', body: { binding: body } }); toast('绑定已保存'); await refreshProject(); renderPane(); } catch (e) { toast(e.message, true); }
    } }, '保存绑定'));
    card.appendChild(bind);
    return card;
  }

  async function videoPane(k) {
    const v = STATE.data.videos;
    const sources = [['original', '原片', v.original], ['lipsync', '口型', v.lipsync], ['enhanced', '增强', v.enhanced], ['final', 'v1 成片', v.final], ['vc', 'v2 成片', v.vc]].filter((s) => s[2]);
    const pref = { lipsync: 'lipsync', enhance: 'enhanced', compose: 'final', v2: 'vc' }[k];
    let cur = sources.find((s) => s[0] === pref) ? pref : (sources[0] || [])[0];
    const video = h('video', { controls: '', preload: 'metadata' });
    const setSrc = (key, keepTime = true) => { const t = video.currentTime, playing = !video.paused; cur = key; video.src = sources.find((s) => s[0] === key)[2]; if (keepTime) video.addEventListener('loadedmetadata', () => { video.currentTime = t; if (playing) video.play(); }, { once: true }); };
    if (cur) setSrc(cur, false);
    const bar = h('div', { class: 'row' }, ...sources.map((s) => h('button', { class: s[0] === cur ? 'primary' : '', onclick: (e) => { setSrc(s[0]); bar.querySelectorAll('button').forEach((b) => b.classList.remove('primary')); e.target.classList.add('primary'); } }, s[1])));
    const card = h('div', { class: 'card' }, h('h3', {}, '预览（切换源保持播放位置）'), bar, video);
    STATE.videoEl = video;
    if (sources.length >= 2) {
      const ab = h('button', { onclick: () => {
        const left = h('video', { controls: '', src: v.original, muted: '' }), right = h('video', { controls: '', src: sources.find((s) => s[0] === cur)[2] });
        left.style.width = right.style.width = '49%';
        const sync = () => { if (Math.abs(left.currentTime - right.currentTime) > 0.15) left.currentTime = right.currentTime; };
        right.addEventListener('timeupdate', sync); right.addEventListener('play', () => left.play()); right.addEventListener('pause', () => left.pause()); right.addEventListener('seeked', sync);
        modal(h('div', {}, h('h3', {}, '并排对比：左 原片（静音） / 右 当前源'), h('div', { class: 'row', style: 'align-items:flex-start' }, left, right)));
      } }, '并排对比');
      bar.appendChild(ab);
    }
    if (k === 'v2') { const s = STATE.data.summary.vc || {}; card.appendChild(h('div', { class: 'small muted' }, s.note || (s.converted != null ? `已转换 ${s.converted} 段，时间轴最大漂移 ${s.max_drift_ms} ms` : ''))); }
    return card;
  }

  async function qcPane() {
    const card = h('div', { class: 'card' });
    let key = 'fit';
    const render = async () => {
      const q = await api(`/api/projects/${STATE.project}/qc?key=${key}`);
      card.innerHTML = '';
      const sw = h('div', { class: 'row' }, h('button', { class: key === 'fit' ? 'primary' : '', onclick: () => { key = 'fit'; render(); } }, 'v1'), h('button', { class: key === 'vc' ? 'primary' : '', onclick: () => { key = 'vc'; render(); } }, 'v2'));
      card.appendChild(h('h3', {}, '质检清单 ', sw));
      if (!q.available) { card.appendChild(h('div', { class: 'muted' }, '尚未生成 QC（运行“质检”步）。')); return; }
      const s = q.summary;
      card.appendChild(h('div', {}, h('span', { class: 'badge ok' }, `PASS ${s.PASS}`), h('span', { class: 'badge warn' }, `WARN ${s.WARN}`), h('span', { class: 'badge bad' }, `FAIL ${s.FAIL}`), ' ', h('span', { class: 'small muted' }, Object.entries(s.reasons || {}).map(([k, v]) => `${k}×${v}`).join('；'))));
      let filter = 'nonpass';
      const tb = h('tbody');
      const draw = () => { tb.innerHTML = ''; for (const r of q.segments) { if (filter === 'nonpass' && r.status === 'PASS') continue; tb.appendChild(h('tr', { class: r.status + ' clickable', onclick: () => seekTo(r.start) }, h('td', { class: 'status' }, r.status), h('td', { class: 'mono small' }, r.tc), h('td', {}, r.speaker || ''), h('td', { class: 'small' }, r.reasons || ''), h('td', {}, r.text_zh || ''))); } };
      card.appendChild(h('div', { class: 'row' }, h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: '', onchange: (e) => { filter = e.target.checked ? 'nonpass' : 'all'; draw(); } }), '只看 WARN/FAIL'), h('span', { class: 'small muted' }, '点击行可定位到下方视频。')));
      const tbl = h('table', {}, h('thead', {}, h('tr', {}, ...['状态', '时间', '说话人', '原因', '译文'].map((t) => h('th', {}, t)))), tb);
      draw();
      card.appendChild(h('div', { style: 'max-height:40vh;overflow:auto' }, tbl));
      const v = STATE.data.videos; const src = key === 'vc' ? (v.vc || v.final) : (v.final || v.lipsync);
      if (src) { const vid = h('video', { controls: '', src, preload: 'metadata' }); STATE.videoEl = vid; card.appendChild(vid); }
    };
    await render();
    return card;
  }
  function seekTo(t) { if (STATE.videoEl) { STATE.videoEl.currentTime = Math.max(0, t - 0.5); STATE.videoEl.play(); } }

  async function deliverPane() {
    const [files, acc] = await Promise.all([api(`/api/projects/${STATE.project}/deliverables`), api(`/api/projects/${STATE.project}/acceptance`)]);
    const card = h('div', { class: 'card' }, h('h3', {}, '交付物与产物'));
    const demos = files.filter((f) => /_dubbed\/.*demo.*\.mp4$/.test(f.name));
    if (demos.length) { const d = h('div', { class: 'grid2' }); for (const f of demos) d.appendChild(h('div', {}, h('div', { class: 'small' }, f.name.split('/').pop()), h('video', { controls: '', src: f.url, preload: 'none' }))); card.appendChild(h('div', {}, h('h3', {}, '演示片段'), d)); }
    const tbl = h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, '文件'), h('th', {}, '大小'), h('th', {}, ''))));
    for (const f of files) tbl.appendChild(h('tr', {}, h('td', {}, f.name), h('td', {}, fmtSize(f.size)), h('td', {}, h('a', { href: f.download }, '下载'), ' ', ['mp4', 'wav', 'jpg', 'png'].includes(f.kind) ? h('a', { href: f.url, target: '_blank' }, '打开') : null)));
    card.appendChild(h('div', { style: 'max-height:40vh;overflow:auto' }, tbl));
    if (acc.text) card.appendChild(h('div', {}, h('h3', {}, '验收结果'), h('pre', { class: 'small', style: 'white-space:pre-wrap' }, acc.text)));
    return card;
  }

  // ── jobs ──
  async function startJob(body) {
    try {
      const r = await api(`/api/projects/${STATE.project}/jobs`, { method: 'POST', body });
      toast('任务已提交'); attachJob(r.job); await refreshProject(); renderPane();
    } catch (e) {
      if (e.status === 409 && e.data && e.data.error === 'gpu_busy') {
        const procs = (e.data.procs || []).map((p) => `${p.pid} ${p.cmd}`).join('\n');
        if (confirm('GPU 被会话外任务占用：\n' + procs + '\n\n仍然提交（会排队但可能争抢显存）？')) return startJob({ ...body, allow_foreign: true });
      } else toast(e.message, true);
    }
  }
  function attachJob(job) {
    STATE.job = job; STATE.seq = 0;
    $('drawer').classList.remove('collapsed'); $('drawer-toggle').textContent = '日志 ▾';
    $('log').textContent = ''; $('btn-cancel').hidden = false;
    $('job-title').textContent = `${job.kind} · ${job.id}`;
    if (STATE.es) STATE.es.close();
    const connect = () => {
      const es = new EventSource(`/api/jobs/${job.id}/events?since=${STATE.seq}`);
      STATE.es = es;
      const onEv = (e) => { const ev = JSON.parse(e.data); if (ev.seq) STATE.seq = ev.seq; return ev; };
      es.addEventListener('log', (e) => { const ev = onEv(e); appendLog(ev.line, ev.banner ? 'banner' : (/Traceback|Error|FAILED/.test(ev.line) ? 'err' : '')); });
      es.addEventListener('progress', (e) => { const ev = onEv(e); const pct = ev.total ? Math.round(100 * ev.done / ev.total) : 0; $('job-bar').style.width = pct + '%'; $('job-progress').textContent = `${ev.step || ''} ${ev.label} ${ev.done}/${ev.total} (${pct}%)`; });
      es.addEventListener('step', (e) => { const ev = onEv(e); $('job-progress').textContent = `${ev.step}: ${statusLabel(ev.status === 'cached' ? 'done' : ev.status)}${ev.took ? ' ' + ev.took + 's' : ''}`; if (ev.status !== 'running') $('job-bar').style.width = '0%'; refreshProject().then(renderToolbar); });
      es.addEventListener('job', (e) => { const ev = onEv(e); if (['done', 'failed', 'cancelled'].includes(ev.status)) { toast(`任务${statusLabel(ev.status)}`, ev.status !== 'done'); STATE.job = null; $('btn-cancel').hidden = true; es.close(); STATE.es = null; STATE.segments = null; refreshProject().then(() => { renderPane(); loadProjects(); }); } });
      es.onerror = () => { es.close(); if (STATE.job && STATE.job.id === job.id) setTimeout(connect, 3000); };
    };
    connect();
  }
  function appendLog(line, cls) {
    const log = $('log'); const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    const el = h('div', { class: cls || '' }, line); log.appendChild(el);
    while (log.childNodes.length > 3000) log.removeChild(log.firstChild);
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
  $('btn-cancel').addEventListener('click', async () => { if (STATE.job && confirm('取消当前任务？')) { try { await api(`/api/jobs/${STATE.job.id}/cancel`, { method: 'POST' }); } catch (e) { toast(e.message, true); } } });
  $('drawer-toggle').addEventListener('click', () => { const d = $('drawer'); d.classList.toggle('collapsed'); $('drawer-toggle').textContent = d.classList.contains('collapsed') ? '日志 ▴' : '日志 ▾'; });
  $('btn-refresh').addEventListener('click', async () => { STATE.segments = null; await refreshProject(); renderPane(); loadProjects(); });

  // ── one-click modal ──
  function modal(content) {
    const root = $('modal-root'); root.innerHTML = '';
    const bg = h('div', { class: 'modal-bg', onclick: (e) => { if (e.target === bg) root.innerHTML = ''; } }, h('div', { class: 'modal' }, content, h('div', { class: 'row', style: 'justify-content:flex-end;margin-top:10px' }, h('button', { onclick: () => root.innerHTML = '' }, '关闭'))));
    root.appendChild(bg);
    return () => root.innerHTML = '';
  }
  $('btn-one-click').addEventListener('click', () => {
    if (!STATE.data) return toast('先选择工程', true);
    const steps = STATE.data.step_order.filter((s) => !['v2', 'deliver'].includes(s));
    const boxes = {};
    const list = h('div', {}, ...steps.map((s) => { const c = h('input', { type: 'checkbox', checked: '' }); boxes[s] = c; return h('label', { class: 'chk' }, c, STATE.data.steps[s].label, h('span', { class: 'small muted' }, `(${statusLabel(STATE.data.steps[s].status)})`)); }));
    const v2 = h('input', { type: 'checkbox', checked: '' }), deliver = h('input', { type: 'checkbox', checked: '' }), force = h('input', { type: 'checkbox' });
    const close = modal(h('div', {}, h('h3', {}, '一键生成'), h('div', { class: 'small muted' }, '已完成且未过期的步骤会自动跳过（除非勾选强制重跑）。'), list,
      h('div', {}, h('label', { class: 'chk' }, v2, '生成 v2 原声音色'), h('label', { class: 'chk' }, deliver, '生成交付包（含演示片段）'), h('label', { class: 'chk' }, force, '强制重跑所选步骤')),
      h('button', { class: 'primary', onclick: () => { close(); startJob({ kind: 'one_click', steps: steps.filter((s) => boxes[s].checked), force: force.checked, with_v2: v2.checked, with_deliver: deliver.checked }); } }, '开始')));
  });

  // ── upload / import ──
  $('btn-upload').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const name = prompt('工程名（字母/数字/下划线/中文）', f.name.replace(/\.[^.]+$/, '')); if (!name) return;
    const CH = 8 * 1024 * 1024;
    try {
      const meta = await api('/api/uploads/init', { method: 'POST', body: { filename: f.name, size: f.size, name } });
      const n = Math.ceil(f.size / CH);
      for (let i = 0; i < n; i++) {
        const blob = f.slice(i * CH, Math.min(f.size, (i + 1) * CH));
        let ok = false;
        for (let attempt = 0; attempt < 3 && !ok; attempt++) { try { const r = await fetch(`/api/uploads/${meta.id}/${i}`, { method: 'PUT', body: blob }); ok = r.ok; } catch (err) { ok = false; } }
        if (!ok) throw new Error(`分块 ${i} 上传失败`);
        $('gpu-note').textContent = `上传 ${name}: ${Math.round(100 * (i + 1) / n)}%`;
      }
      const r = await api(`/api/uploads/${meta.id}/finalize`, { method: 'POST' });
      toast(`已上传 ${r.name}（${fmtSize(r.size)}）`); $('gpu-note').textContent = '';
      await loadProjects(); openProject(r.name);
    } catch (err) { toast(err.message, true); }
    e.target.value = '';
  });
  $('btn-import').addEventListener('click', async () => {
    const files = await api('/api/inputs');
    const free = files.filter((f) => !f.has_project);
    if (!free.length) return toast('inputs/ 里没有未登记的视频', true);
    const sel = h('select', {}, ...free.map((f) => h('option', { value: f.file }, `${f.file} (${fmtSize(f.size)})`)));
    const close = modal(h('div', {}, h('h3', {}, '导入 inputs/ 中的视频'), sel, h('button', { class: 'primary', style: 'margin-left:8px', onclick: async () => { try { const r = await api('/api/projects/import', { method: 'POST', body: { file: sel.value } }); close(); toast('已登记 ' + r.name); await loadProjects(); openProject(r.name); } catch (e) { toast(e.message, true); } } }, '导入')));
  });

  // global events: keep the project list dots fresh
  const ges = new EventSource('/api/events');
  ges.addEventListener('job', () => loadProjects());
  setInterval(() => { if (STATE.project && !STATE.job) refreshProject().catch(() => {}); }, 30000);

  loadProjects().then(() => { const last = localStorage.getItem('ai-movie-project'); if (last) openProject(last); });
  const _open = openProject; openProject = (n) => { try { localStorage.setItem('ai-movie-project', n); } catch (e) {} return _open(n); };
})();
