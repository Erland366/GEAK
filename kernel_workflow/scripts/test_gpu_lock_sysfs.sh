#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# gpu_lock.sh must map GPU ids to sysfs through KFD topology (not render-minor
# arithmetic), honor numeric ROCR_VISIBLE_DEVICES, and report unknown -- never
# idle -- when a GPU cannot be mapped or its telemetry cannot be read.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$SCRIPT_DIR/gpu_lock.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
root="$tmp/sys"
nodes="$root/class/kfd/kfd/topology/nodes"

# node <name> <simd_count> <render_minor> <location_id>
node() {
  mkdir -p "$nodes/$1"
  printf 'simd_count %s\ndrm_render_minor %s\nlocation_id %s\ndomain 0\n' "$2" "$3" "$4" > "$nodes/$1/properties"
}
# render <minor> <pci_bdf> <busy_percent> <vram_used_bytes>
render() {
  mkdir -p "$root/devices/$2" "$root/class/drm/renderD$1"
  ln -s "$root/devices/$2" "$root/class/drm/renderD$1/device"
  echo "$3" > "$root/devices/$2/gpu_busy_percent"
  echo "$4" > "$root/devices/$2/mem_info_vram_used"
}

# Two CPU nodes, then GPU nodes "2" and "10" with consecutive render minors (as on
# MI210 hosts). Node "10" sorts before "2" lexically; ROCr order is numeric.
node 0 0 0 0
node 1 0 0 0
node 2 416 128 18176      # 0000:47:00.0
node 10 416 129 18944     # 0000:4a:00.0
render 128 0000:47:00.0 0 300000000
render 129 0000:4a:00.0 90 2000000000

run() {  # run <VAR=value...> -- <function> <args...>; prints rc
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  env -u ROCR_VISIBLE_DEVICES GEAK_SYSFS_ROOT="$root" "${vars[@]}" bash -c '
    set -uo pipefail
    eval "$(sed -n "/^GEAK_SYSFS_ROOT=/p; /^_gpu_sysfs_dev()/,/^}/p; /^_gpu_is_idle()/,/^}/p" "'"$LOCK"'")"
    "$0" "$@" >/dev/null 2>&1; echo $?
  ' "$@"
}
expect() { [ "$1" = "$2" ] || fail "$3 (expected rc $2, got $1)"; }

dev0="$(GEAK_SYSFS_ROOT="$root" bash -c '
  eval "$(sed -n "/^GEAK_SYSFS_ROOT=/p; /^_gpu_sysfs_dev()/,/^}/p" "'"$LOCK"'")"
  _gpu_sysfs_dev 0')"
[ "${dev0##*/}" = "0000:47:00.0" ] || fail "GPU 0 should map to 0000:47:00.0 (got $dev0)"
dev1="$(GEAK_SYSFS_ROOT="$root" bash -c '
  eval "$(sed -n "/^GEAK_SYSFS_ROOT=/p; /^_gpu_sysfs_dev()/,/^}/p" "'"$LOCK"'")"
  _gpu_sysfs_dev 1')"
[ "${dev1##*/}" = "0000:4a:00.0" ] || fail "GPU 1 should map to 0000:4a:00.0 via numeric node order (got $dev1)"

expect "$(run -- _gpu_is_idle 0)" 0 "idle GPU 0"
expect "$(run -- _gpu_is_idle 1)" 1 "busy GPU 1 (renderD129, not stride-eight renderD136)"
expect "$(run ROCR_VISIBLE_DEVICES=1 -- _gpu_is_idle 0)" 1 "logical 0 under ROCR_VISIBLE_DEVICES=1 is physical 1"
expect "$(run ROCR_VISIBLE_DEVICES=1 -- _gpu_is_idle 1)" 2 "logical 1 outside a one-GPU ROCR mask"
expect "$(run ROCR_VISIBLE_DEVICES=GPU-a2933f290499b2eb -- _gpu_is_idle 0)" 2 "UUID ROCR mask is unknown"
expect "$(run -- _gpu_is_idle 5)" 2 "GPU id without a KFD node"
expect "$(run -- _gpu_is_idle x)" 2 "non-numeric GPU id"

lockfile() {  # lockfile <ROCR mask or ""> <id>
  env -u ROCR_VISIBLE_DEVICES ${1:+ROCR_VISIBLE_DEVICES=$1} GEAK_SYSFS_ROOT="$root" bash -c '
    eval "$(sed -n "/^LOCK_DIR=/p; /^GEAK_SYSFS_ROOT=/p; /^_gpu_sysfs_dev()/,/^}/p; /^_gpu_lock_file()/,/^}/p" "'"$LOCK"'")"
    _gpu_lock_file "$0"' "$2"
}
[ "$(lockfile "" 1)" = "/tmp/team_gpu_locks/gpu_0000:4a:00.0.lock" ] || fail "lock file must be named by PCI address"
[ "$(lockfile 1 0)" = "$(lockfile "" 1)" ] || fail "one physical GPU must get one lock file under any ROCR mask"
[ "$(lockfile "" 0)" != "$(lockfile 1 0)" ] || fail "different physical GPUs must not share a lock file"
lockfile GPU-a2933f290499b2eb 0 >/dev/null 2>&1 && fail "an unmappable GPU must not get a lock file"
[ "$(GEAK_GPU_REQUIRE_IDLE=0 lockfile "" 9876 2>/dev/null)" = "/tmp/team_gpu_locks/gpu_unmapped_9876.lock" ] \
  || fail "an explicit inspection opt-out may lock an unmappable id under the unmapped name only"

rm "$root/devices/0000:47:00.0/gpu_busy_percent"
expect "$(run -- _gpu_is_idle 0)" 2 "missing busy telemetry is unknown, not idle"
echo 0 > "$root/devices/0000:47:00.0/gpu_busy_percent"

node 2 416 128 2048       # location now says 0000:08:00.0, render node is 0000:47:00.0
expect "$(run -- _gpu_is_idle 0)" 2 "KFD/DRM PCI mismatch is unknown"

grep -q 'KERNEL_ENV_SKIP_ENUM_REAP:-1}' "$LOCK" || fail "orphan enumerator reaping must be opt-in"
grep -qF 'renderD$((128 + 8' "$LOCK" && fail "stride-eight render-minor arithmetic is back"

echo "PASS: gpu_lock sysfs mapping"
