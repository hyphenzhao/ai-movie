# Web 工作台与远程访问（2026-09-12）

## 是什么

`ai_movie/web/` 是桌面 Tk 程序的 Web 版：同一条无头流水线（`scripts/run_pipeline.py` 及 v2/交付脚本）通过 HTTP 驱动，
可以逐步或一键生成、实时看日志与进度、修改译文/说话人/术语表/人脸绑定、在线预览原片/口型/增强/成片与逐段音频、
查看 QC 清单并点击定位、下载交付物。运行在局域网主机（ROCm 计算机）上，局域网内免登录；异地通过 VPS 反向代理访问，
边缘由 Authelia 做用户名 + 密码 + 手机验证器 TOTP。

## 访问方式

| 场景 | 地址 | 登录 |
|---|---|---|
| 局域网 | `http://192.168.20.1:8000/` 或 `http://10.200.58.191:8000/` | 无 |
| 异地 | `https://ai-movie.hfnjc.net/` | Authelia：用户名 `haifeng` + 密码 + TOTP（Okta Verify / Google Authenticator） |

远程拓扑：浏览器 → Cloudflare（当前域名走了 CF 代理）→ VPS 64.176.52.137 Caddy（Let's Encrypt 证书）→ `forward_auth` Authelia
（`127.0.0.1:9091`，门户在 `/authelia/`）→ `reverse_proxy 127.0.0.1:18000` → SSH 反向隧道 → 本机 `127.0.0.1:8000`。

## 本机服务（systemd --user，随开机启动，Linger 已开）

```
systemctl --user status ai-movie-web      # FastAPI/uvicorn :8000，日志 workspace/web.log
systemctl --user status ai-movie-tunnel   # ssh -R 127.0.0.1:18000:127.0.0.1:8000 tunnel@VPS，10 s 自动重连
```
Web 服务 `KillMode=process`：重启 Web 不会杀正在跑的流水线，任务按 pid 重新接管并继续从日志文件推送。

## 后端设计要点

- **不复制流水线逻辑**：每个任务 = 一组子进程命令（`run_pipeline.py --steps …`、`auto_select_refs.py`、`run_vc_version.py --refs-json`、
  `eval_pipeline.py`、`deliver.py`），独立进程组，`AI_MOVIE_JSON_LOG=1` 让 `run_pipeline.log()` 输出 NDJSON，tail 线程解析成
  `log / progress / step / job` 事件经 SSE 推给浏览器（`/api/jobs/{id}/events?since=`，15 s ping，断线续传）。
- **单任务锁**：GPU 是统一内存，同一时刻只跑一个任务；终端里另起的流水线进程（`pgrep`）会让新任务返回 409 `gpu_busy`，UI 可选择仍然排队。
- **状态**：`run_pipeline.py --status-json` 用工程保存的选项（`workspace/<name>/web/options.json`）算每步 `valid/stale/missing/…`，
  Web 再按 `STEP_DEPS` 拓扑推导 `locked/ready/done/stale/running/failed`，并加两个伪步骤 `v2`、`deliver`。
- **编辑会让下游过期**：段文本不在指纹里，所以编辑会把 `state["_edits"][stage]` 加一并进入该步指纹的 `extra`
  （`run_pipeline.restamp_after_edit`）；被保留的步骤（改说话人时的 glossary/translate）刷新 `up` 保持有效，其余下游自然 STALE。
  改译文 → tts 起过期；改说话人/性别（`diarize.assign_speaker_for_gender`，无同性别说话人则新建 `S<k>`）→ tts 起过期、翻译保留；
  术语表 → translate 过期，或“仅应用到现有译文”做字符串替换；人脸绑定 → 选项 `--faces-bind S0=3,S1=none`（faces 过期）。
  每次编辑前备份 `web/state_backups/`（保留 20 份）。任务运行/排队中的工程拒绝编辑（409）。
- **媒体**：`/api/media?p=<绝对路径>` 只允许 `workspace/ inputs/ deliver/` 下白名单扩展名，`FileResponse` 原生 Range；下载带 UTF-8 文件名。
- **上传**：8 MB 分块（`/api/uploads/init|{id}/{n}|finalize`）以穿过隧道与 Cloudflare 的 100 MB 单请求上限；也可 scp 到 `inputs/` 后“导入”。

## 前端

`ai_movie/web/static/{index.html,app.js,app.css}`，原生 JS + CSS，不引用任何外部资源（局域网可能屏蔽 CDN）。
左侧工程列表；顶部 17 个步骤按钮按状态着色；每步选项面板与 CLI 一一对应；底部日志/进度抽屉；一键生成弹窗（各步复选 + v2 + 交付 + 强制）。

## 测试

```
bash scripts/web_smoke.sh [http://host:8000] [project]   # health/状态/Range/穿越/SSE/下载/首页
.venv/bin/python tests/web/test_edits.py                  # 编辑语义与指纹过期（合成工作区）
node scripts/web_check.js                                  # app.js 语法、id 引用、无外链、API 路径与路由一致
```
实测：三项全部通过；远程链路用密码 + TOTP 登录后 `/api/health`、`/`、`/static/app.js`、`/api/projects` 均 200，Range 请求 206。

## VPS 配置摘要（Ubuntu 22.04，1 vCPU / 951 MB）

- 用户：`ops`（sudo，密钥登录）、`tunnel`（仅允许 `-R 127.0.0.1:18000`，`ForceCommand /bin/false`）、`root` 仅密钥；
  `PasswordAuthentication no`。加了 1 GB swap。ufw：22/80/443（另有此前已存在的 8000 与 vsftpd，未动）。
- Caddy（apt 官方源）：`/etc/caddy/Caddyfile`，`@authelia path /authelia*` 直通门户，其余 `forward_auth` 后代理到隧道端口；`flush_interval -1` 保 SSE。
- Authelia v4.39.26（静态二进制，systemd，`MemoryMax=200M`）：`/etc/authelia/configuration.yml`（子路径门户、argon2id 文件后端、
  `two_factor`、3 次错误封 10 分钟、sqlite、文件通知器）、`/etc/authelia/users_database.yml`。
- 域名 `ai-movie.hfnjc.net` 目前在 Cloudflare 橙云后面：ACME HTTP-01 仍成功签发；Cloudflare 免费版单请求 100 MB（已用分块上传规避）、
  会拦截非浏览器 UA（curl/urllib 需带浏览器 User-Agent）。改为“仅 DNS”即可去掉这些限制，Caddy 配置无需改动。

## 常用操作

- 新增 Web 用户：在 VPS 上 `authelia crypto hash generate argon2 --random -p 1` 得到密码与摘要，写入 `users_database.yml`（自动热加载）；
  `authelia storage user totp generate <user> --config /etc/authelia/configuration.yml --path /root/<user>.png` 生成二维码，发给用户后 `shred`。
- 看远程访问日志：`journalctl -u caddy -f`、`/var/log/authelia/authelia.log`。
- 隧道断了：本机 `systemctl --user restart ai-movie-tunnel`；VPS 上 `curl 127.0.0.1:18000/api/health` 验证。
