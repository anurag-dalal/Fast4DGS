
#!/usr/bin/env python3

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Device:
	user: str
	ip: str
	password: str


DEVICES: List[Device] = [
	Device("nodeone", "10.0.0.1", "NodeOne"),
	Device("nodetwo", "10.0.0.2", "NodeTwo"),
	Device("nodethree", "10.0.0.3", "NodeThree"),
	Device("nodefour", "10.0.0.4", "NodeFour"),
]


REMOTE_CAPTURE_DIR = "/tmp/stream_captures"
LOCAL_CAPTURE_DIR = os.path.expanduser("~/stream_captures")
VIDEO_IDXS = list(range(6))


def _require_tool(name: str) -> None:
	if shutil.which(name) is None:
		raise RuntimeError(f"Missing required tool '{name}' in PATH")


def _run(cmd: List[str], *, timeout_s: int, check: bool = True) -> subprocess.CompletedProcess:
	return subprocess.run(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		timeout=timeout_s,
		check=check,
	)


def _ssh(device: Device, remote_cmd: str, *, timeout_s: int) -> subprocess.CompletedProcess:
	cmd = [
		"sshpass",
		"-p",
		device.password,
		"ssh",
		"-o",
		"StrictHostKeyChecking=no",
		f"{device.user}@{device.ip}",
		remote_cmd,
	]
	return _run(cmd, timeout_s=timeout_s, check=False)


def _scp_from(device: Device, remote_path: str, local_path: str, *, timeout_s: int) -> subprocess.CompletedProcess:
	cmd = [
		"sshpass",
		"-p",
		device.password,
		"scp",
		"-o",
		"StrictHostKeyChecking=no",
		f"{device.user}@{device.ip}:{remote_path}",
		local_path,
	]
	return _run(cmd, timeout_s=timeout_s, check=False)


def _build_remote_script(device: Device) -> str:
	# Use bash on the remote host; keep it robust (continue on per-camera failures).
	# Capture pipeline: single buffer from v4l2 -> convert -> png -> file.
	# -q reduces gst-launch logging.
	pass_q = shlex.quote(device.password)
	remote_dir_q = shlex.quote(REMOTE_CAPTURE_DIR)

	lines = [
		"set -euo pipefail",
		f"mkdir -p {remote_dir_q}",
		f"rm -f {remote_dir_q}/cam*.png || true",
		# Stop running streamers (mirrors remote_shutdown.sh but without shutdown)
		f"echo {pass_q} | sudo -S pkill -9 gst-launch-1.0 >/dev/null 2>&1 || true",
	]

	for vid in VIDEO_IDXS:
		dev_path_q = shlex.quote(f"/dev/video{vid}")
		out_path_q = shlex.quote(f"{REMOTE_CAPTURE_DIR}/cam{vid}.png")
		bang = "'!'"
		caps = shlex.quote("video/x-raw,width=(int)1920,height=(int)1080")
		# Keep each camera independent so one failure doesn't abort the whole run.
		# Force the *output PNG* to 1920x1080 using videoscale, even if the input
		# negotiates a different size.
		lines.append(
			"{ "
			f"echo {pass_q} | sudo -S v4l2-ctl --device={dev_path_q} "
			"--set-fmt-video=width=1920,height=1080,pixelformat=UYVY >/dev/null 2>&1 || true; "
			f"gst-launch-1.0 -q v4l2src device={dev_path_q} num-buffers=1 "
			f"{bang} videoconvert {bang} videoscale {bang} {caps} {bang} pngenc {bang} filesink location={out_path_q}; "
			"} >/dev/null 2>&1 || true"
		)

	# Print what was produced for easier debugging.
	lines.append(f"ls -1 {remote_dir_q} || true")

	script = "\n".join(lines)
	# Avoid sourcing remote profile/rc files (some nodes may have incompatible shell init).
	return "bash --noprofile --norc -c " + shlex.quote(script)


def main() -> int:
	try:
		_require_tool("sshpass")
		_require_tool("ssh")
		_require_tool("scp")
	except Exception as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		return 2

	os.makedirs(LOCAL_CAPTURE_DIR, exist_ok=True)

	total_expected = len(DEVICES) * len(VIDEO_IDXS)
	print(f"Saving {total_expected} images to {LOCAL_CAPTURE_DIR}")

	for node_idx, device in enumerate(DEVICES):
		print("---------------------------------------")
		print(f"[{node_idx+1}/{len(DEVICES)}] Connecting to {device.user}@{device.ip} ...")

		remote_cmd = _build_remote_script(device)
		try:
			res = _ssh(device, remote_cmd, timeout_s=180)
		except subprocess.TimeoutExpired:
			print(f"ERROR: SSH timed out for {device.ip}")
			continue

		if res.returncode != 0:
			print(f"WARN: Remote command non-zero on {device.ip} (rc={res.returncode})")
		if res.stdout:
			print(res.stdout.strip())

		for vid in VIDEO_IDXS:
			remote_file = f"{REMOTE_CAPTURE_DIR}/cam{vid}.png"
			cam_global_idx = node_idx * len(VIDEO_IDXS) + vid
			local_file = os.path.join(LOCAL_CAPTURE_DIR, f"cam{cam_global_idx}.png")

			try:
				scp_res = _scp_from(device, remote_file, local_file, timeout_s=60)
			except subprocess.TimeoutExpired:
				print(f"ERROR: SCP timed out: {device.ip}:{remote_file}")
				continue

			if scp_res.returncode == 0:
				print(f"OK: {device.ip} /dev/video{vid} -> cam{cam_global_idx}.png")
			else:
				msg = (scp_res.stdout or "").strip()
				print(
					f"FAIL: {device.ip} /dev/video{vid} (rc={scp_res.returncode})"
					+ (f"\n{msg}" if msg else "")
				)

	print("---------------------------------------")
	print("Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

