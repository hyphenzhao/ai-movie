#!/usr/bin/env python3
"""Tiny HTTP proxy that sends everything out of one chosen network interface.

    python scripts/side_proxy.py --bind-ip 10.200.53.85 --listen 127.0.0.1:18080

Why: the VPN's policy routing sends all traffic into its tunnel; a source
rule (``from <side-ip> lookup 300``) lets packets that originate from the
side adapter's own address bypass it.  Programs that cannot bind an
interface themselves (browsers, pip/uv, huggingface-cli) get there through
this proxy: it accepts ``CONNECT`` (https) and absolute-URI requests (http)
on localhost and opens the outbound socket bound to *bind-ip*.

Only listens on localhost; no authentication; not for exposure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from urllib.parse import urlsplit

BIND_IP = "127.0.0.1"


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:                               # noqa: BLE001
            pass


async def open_out(host: str, port: int):
    return await asyncio.wait_for(asyncio.open_connection(host, port, local_addr=(BIND_IP, 0)), 20)


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
    except Exception:                                   # noqa: BLE001
        writer.close(); return
    line, _, rest = head.partition(b"\r\n")
    try:
        method, target, version = line.decode("latin-1").split(" ", 2)
    except ValueError:
        writer.close(); return
    try:
        if method.upper() == "CONNECT":
            host, _, port = target.rpartition(":")
            r2, w2 = await open_out(host, int(port or 443))
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
        else:
            u = urlsplit(target)
            if not u.hostname:
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n"); await writer.drain(); writer.close(); return
            r2, w2 = await open_out(u.hostname, u.port or 80)
            path = (u.path or "/") + (f"?{u.query}" if u.query else "")
            hdrs = [h for h in rest.split(b"\r\n") if h and not h.lower().startswith(b"proxy-")]
            w2.write(f"{method} {path} {version}\r\n".encode("latin-1") + b"\r\n".join(hdrs) + b"\r\n\r\n")
            await w2.drain()
    except Exception as exc:                            # noqa: BLE001
        try:
            writer.write(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{exc}".encode()); await writer.drain()
        except Exception:                               # noqa: BLE001
            pass
        writer.close(); return
    await asyncio.gather(pipe(reader, w2), pipe(r2, writer))


async def main_async(listen: str) -> None:
    host, _, port = listen.rpartition(":")
    server = await asyncio.start_server(handle, host or "127.0.0.1", int(port))
    print(f"side proxy on {listen} → outbound via {BIND_IP}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> int:
    global BIND_IP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind-ip", required=True, help="the side adapter's own IPv4 address")
    ap.add_argument("--listen", default="127.0.0.1:18080")
    args = ap.parse_args()
    BIND_IP = args.bind_ip
    try:
        asyncio.run(main_async(args.listen))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
