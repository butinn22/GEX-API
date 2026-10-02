"""Create (or refresh) the "GEX Trading API" desktop shortcut.

We write the ``.lnk`` ourselves rather than driving ``WScript.Shell``: COM
shortcut creation is blocked in this environment, and the Shell Link format
(MS-SHLLINK) is a simple, fully documented binary layout. The upside is that the
shortcut is reproducible from source and can be regenerated if the repo moves:

    .venv/Scripts/python.exe scripts/make_shortcut.py            # create/update
    .venv/Scripts/python.exe scripts/make_shortcut.py --verify   # parse it back
    .venv/Scripts/python.exe scripts/make_shortcut.py --remove

Structure written (all little-endian):

    [ShellLinkHeader 76B] [LinkInfo] [StringData NAME, WORKING_DIR, ICON_LOCATION]

The shortcut deliberately carries **no** LinkTargetIDList and no RELATIVE_PATH:
an IDList would require encoding shell-namespace item IDs, and RELATIVE_PATH is
defined *relative to the .lnk file itself* — which breaks when the repo lives on
a different drive from the desktop (``os.path.relpath`` raises across drives).
An absolute ``LocalBasePath`` in LinkInfo is authoritative and sufficient.
"""
from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run.bat"
ICON = ROOT / "trading" / "static" / "favicon.ico"
SHORTCUT_NAME = "GEX Trading API"
DESCRIPTION = "Start the GEX Trading API (portfolio backtest console) and open the browser"

# ── MS-SHLLINK constants ───────────────────────────────────────────────

HEADER_SIZE = 0x4C  # 76

#: LinkCLSID {00021401-0000-0000-C000-000000000046} in on-the-wire order.
CLSID_SHELL_LINK = bytes.fromhex("0114020000000000c000000000000046")

HAS_LINK_INFO = 0x00000002
HAS_NAME = 0x00000004
HAS_WORKING_DIR = 0x00000010
HAS_ICON_LOCATION = 0x00000040
IS_UNICODE = 0x00000080

FILE_ATTRIBUTE_ARCHIVE = 0x00000020
SW_SHOWNORMAL = 1

LINK_INFO_HEADER_SIZE = 0x1C  # 28 — no Unicode path offsets, so ANSI only
VOLUME_ID_DRIVE_FIXED = 3


def _desktop_dir() -> Path:
    """The real desktop path, honouring OneDrive/domain redirection."""
    buf = ctypes.create_unicode_buffer(260)
    CSIDL_DESKTOPDIRECTORY = 0x0010
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOPDIRECTORY, None, 0, buf)
    return Path(buf.value)


def _utf16_string(value: str) -> bytes:
    """A StringData entry: uint16 character count + UTF-16LE (no terminator)."""
    return struct.pack("<H", len(value)) + value.encode("utf-16-le")


def _link_info(local_base_path: str) -> bytes:
    """LinkInfo with a VolumeID and an absolute ANSI LocalBasePath."""
    volume_id = struct.pack("<IIII", 17, VOLUME_ID_DRIVE_FIXED, 0, 0x10) + b"\x00"
    base = local_base_path.encode("cp1252", "replace") + b"\x00"
    suffix = b"\x00"

    volume_id_offset = LINK_INFO_HEADER_SIZE
    local_base_path_offset = volume_id_offset + len(volume_id)
    common_path_suffix_offset = local_base_path_offset + len(base)
    total = common_path_suffix_offset + len(suffix)

    header = struct.pack(
        "<IIIIIII",
        total,
        LINK_INFO_HEADER_SIZE,
        0x00000001,  # VolumeIDAndLocalBasePath
        volume_id_offset,
        local_base_path_offset,
        0,  # CommonNetworkRelativeLinkOffset
        common_path_suffix_offset,
    )
    assert len(header) == LINK_INFO_HEADER_SIZE, len(header)
    assert len(header) + len(volume_id) + len(base) + len(suffix) == total
    return header + volume_id + base + suffix


def build_shortcut(target: Path, working_dir: Path, icon: Path | None, name: str) -> bytes:
    link_flags = HAS_LINK_INFO | HAS_NAME | HAS_WORKING_DIR | IS_UNICODE
    if icon is not None:
        link_flags |= HAS_ICON_LOCATION

    header = (
        struct.pack("<I", HEADER_SIZE)
        + CLSID_SHELL_LINK
        + struct.pack(
            # LinkFlags, FileAttributes, 3x FILETIME, FileSize, IconIndex,
            # ShowCommand, HotKey, Reserved1..3  -> 12 fields, 56 bytes
            "<IIQQQIiIHHII",  # 12 fields = 56 bytes (see table above)
            link_flags,
            FILE_ATTRIBUTE_ARCHIVE,
            0,  # CreationTime — let Explorer stamp it
            0,  # AccessTime    — 0 means "not tracked", avoids a fake timestamp
            0,  # WriteTime
            0,  # FileSize      — only meaningful for file targets with an IDList
            0,  # IconIndex
            SW_SHOWNORMAL,
            0,  # HotKey
            0,  # Reserved1
            0,  # Reserved2
            0,  # Reserved3
        )
    )
    assert len(header) == HEADER_SIZE, len(header)

    strings = _utf16_string(name) + _utf16_string(str(working_dir))
    if icon is not None:
        strings += _utf16_string(str(icon))

    return header + _link_info(str(target)) + strings


def parse_shortcut(data: bytes) -> dict:
    """Independent read-back, so ``--verify`` doesn't just re-trust the writer."""
    if len(data) < HEADER_SIZE:
        raise ValueError("file is too small to be a Shell Link")
    header_size = struct.unpack_from("<I", data, 0)[0]
    if header_size != HEADER_SIZE:
        raise ValueError(f"bad HeaderSize {header_size:#x}")
    if data[4:20] != CLSID_SHELL_LINK:
        raise ValueError("bad LinkCLSID")

    flags = struct.unpack_from("<I", data, 20)[0]
    link_info_size = 0
    if flags & HAS_LINK_INFO:
        link_info_size = struct.unpack_from("<I", data, HEADER_SIZE)[0]
        header_len = struct.unpack_from("<I", data, HEADER_SIZE + 4)[0]
        base_off = struct.unpack_from("<I", data, HEADER_SIZE + 16)[0]
        start = HEADER_SIZE + base_off
        end = data.index(b"\x00", start)
        target = data[start:end].decode("cp1252")

    out: dict = {"flags": flags, "link_info_size": link_info_size, "target": target}

    # StringData entries follow LinkInfo, in the order their flags dictate.
    pos = HEADER_SIZE + link_info_size
    for bit, key in (
        (HAS_NAME, "name"),
        (HAS_WORKING_DIR, "working_dir"),
        (HAS_ICON_LOCATION, "icon"),
    ):
        if not flags & bit:
            continue
        count = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        if flags & IS_UNICODE:
            out[key] = data[pos : pos + count * 2].decode("utf-16-le")
            pos += count * 2
        else:
            out[key] = data[pos : pos + count].decode("cp1252")
            pos += count
    out["trailing_bytes"] = len(data) - pos
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--desktop", type=Path, default=None, help="override the desktop dir")
    parser.add_argument("--name", default=SHORTCUT_NAME, help="shortcut base name")
    parser.add_argument("--verify", action="store_true", help="parse the shortcut back and report")
    parser.add_argument("--remove", action="store_true", help="delete the shortcut")
    args = parser.parse_args(argv)

    desktop = args.desktop or _desktop_dir()
    lnk = desktop / f"{args.name}.lnk"

    if args.remove:
        if lnk.exists():
            lnk.unlink()
            print(f"removed {lnk}")
        else:
            print(f"nothing to remove at {lnk}")
        return 0

    if args.verify:
        if not lnk.exists():
            print(f"MISSING {lnk}")
            return 1
        info = parse_shortcut(lnk.read_bytes())
        print(f"shortcut : {lnk}")
        print(f"  target      : {info['target']}")
        print(f"  working_dir : {info.get('working_dir')}")
        print(f"  icon        : {info.get('icon')}")
        print(f"  name        : {info.get('name')}")
        ok = (
            Path(info["target"]) == LAUNCHER
            and Path(info.get("working_dir", "")) == ROOT
            and info.get("icon") == str(ICON)
            and info["trailing_bytes"] == 0
        )
        print(f"  VERIFY      : {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    for required in (LAUNCHER, ICON):
        if not required.exists():
            print(f"missing {required} — run scripts/make_icon.py first?", file=sys.stderr)
            return 1

    lnk.write_bytes(build_shortcut(LAUNCHER, ROOT, ICON, args.name))
    print(f"wrote {lnk} ({lnk.stat().st_size} bytes)")
    print(f"  -> {LAUNCHER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
