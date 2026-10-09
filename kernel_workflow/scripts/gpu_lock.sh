#!/bin/bash
# GPU lock + per-workspace build isolation wrapper.
# Usage:  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>
#
# Run EVERY kernel command (compile / correctness / benchmark / profile) through this wrapper,
# invoked from inside the workspace directory. It does three generic things — none kernel-specific:
#
#  1. flock per GPU id  -> multiple engineers can share GPUs safely (exclusive during the command).
#  2. TORCH_EXTENSIONS_DIR = <workspace>/.torch_ext  -> isolates the torch cpp_extension build cache
#     PER WORKSPACE. Without this, torch.utils.cpp_extension.load(name=...) compiles every engineer's
#     DIFFERENT source into ONE global cache (~/.cache/torch_extensions/...), which both serializes
#     all parallel compiles on a single global lock AND lets one engineer benchmark another's .so.
#     Deriving it from $PWD makes each isolated workspace get its own cache. (Honors a caller-set
#     TORCH_EXTENSIONS_DIR if already exported.)
#  3. PYTORCH_ROCM_ARCH = the local GPU's gfx arch only -> avoids compiling for ~9 architectures
#     (huge compile speedup). Runtime perf and correctness are unaffected (the kernel runs on the
#     local arch either way). Honors a caller-set PYTORCH_ROCM_ARCH if already exported.

set -euo pipefail

# The usage line deliberately shows NO concrete ids. It used to read "e.g. 0,1,2,3", and that
# example was copied verbatim into real commands by agents improvising one-off checks -- the
# literal string "0,1,2,3" turned up in 15 invocations from runs that had been allocated neither
# GPU 2 nor 3. An example in a usage line gets read as a default; here the value is never
# defaultable, so it names no ids.
GPU_SPEC="${1:?Usage: gpu_lock.sh <gpu_id|pool> <command...>   (pool = comma list of the GPUs THIS run was allocated)}"
shift

# Source provenance is checked before executing any command and again on exit:
# generated builders can recreate a stale overlay during the command itself.
# Exit 86 invalidates all output from that invocation, even if it printed PASS.
SOURCE_GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/workspace_sources.py"
SOURCE_WORKSPACE="$PWD"
python3 "$SOURCE_GUARD" check --workspace "$SOURCE_WORKSPACE"
_source_audit_size=$(stat -c %s "$SOURCE_WORKSPACE/.geak/invalid_measurements.jsonl" 2>/dev/null || echo 0)
_check_sources_on_exit() {
    local command_status=$?
    trap - EXIT
    python3 "$SOURCE_GUARD" check --workspace "$SOURCE_WORKSPACE" || exit 86
    # A builder may catch a check-input failure and continue, or repair the link
    # later in the same command. Neither makes that invocation's output valid.
    local audit_size
    audit_size=$(stat -c %s "$SOURCE_WORKSPACE/.geak/invalid_measurements.jsonl" 2>/dev/null || echo 0)
    if [ "$audit_size" != "$_source_audit_size" ]; then
        echo "GEAK_SOURCE_INVALID: source validation failed during this command; discard all measurement output." >&2
        exit 86
    fi
    exit "$command_status"
}
trap _check_sources_on_exit EXIT

LOCK_DIR="/tmp/team_gpu_locks"
mkdir -p "$LOCK_DIR"

# ---- Allocation fence ---------------------------------------------------------------------------
# GEAK_GPU_ALLOWED (comma list) is the set of GPUs the CALLER was actually allocated. When it is
# set, a <gpu_spec> naming anything outside it is refused. Unset = no fence, so every existing
# caller is unaffected.
#
# Why this exists: the spec arrives as an argv string written by whoever composes the command, and
# in an agent-driven workflow that author is a model. Agents reproduce the usage line's example
# verbatim when improvising a check outside the workflow's normal call sites. The idleness test
# below catches the common case -- a busy foreign card gets skipped -- but it is the wrong
# instrument: it asks "is this GPU free?" when the question is "is this GPU MINE?". A neighbour's
# momentarily-idle card passes idleness and fails ownership, and a run labelled "2 GPUs" silently
# becomes a 3-GPU run, which invalidates the measurement rather than merely slowing it.
#
# Enforced here because this wrapper is the single chokepoint: every compile/correctness/benchmark/
# profile command goes through it by contract, so one check also covers call sites that do not exist
# yet. A hard error rather than a silent intersection -- a caller asking for a GPU it does not own
# has a bug, and quietly running elsewhere would hide it while still producing a number.
if [ -n "${GEAK_GPU_ALLOWED:-}" ]; then
    _bad=""
    for _r in $(echo "$GPU_SPEC" | tr ',' ' '); do
        case ",${GEAK_GPU_ALLOWED}," in *",${_r},"*) ;; *) _bad="$_bad $_r" ;; esac
    done
    if [ -n "$_bad" ]; then
        echo "ERROR: gpu_lock.sh asked for GPU(s)${_bad} but this run is allocated only [${GEAK_GPU_ALLOWED}]." >&2
        echo "       Use the allocated set: gpu_lock.sh ${GEAK_GPU_ALLOWED} <command...>" >&2
        exit 1
    fi
fi

# ---- GPU selection ------------------------------------------------------------------------------
# <gpu_spec> is either a single id ("2", the historical contract, unchanged) or a POOL ("0,1,2,3").
# With a pool we do not pre-assign a GPU: we take the first one that is BOTH unlocked AND idle,
# retrying until one frees. Static pre-assignment (the old `GPU_LIST[i % n]`) cannot self-balance,
# because the binding is chosen before anyone knows how long a job runs -- in a real 4-GPU run that
# left one GPU with 194 lock calls and another with 0.
#
# The pool is whatever the caller passes, so this scales to 1, 2, 4, 8 ... GPUs with no code change.
#
# IDLENESS (GEAK_GPU_REQUIRE_IDLE=1, the default) is checked against the KERNEL DRIVER via sysfs,
# not against our own locks -- a foreign tenant's job is invisible to flock but will still corrupt
# timings. Measured on an idle MI350X: gpu_busy_percent=0, mem_info_vram_used=284MB; under load:
# 100% / 1837MB. amd-smi is deliberately NOT used here: it returns EMPTY output while another
# process holds the GPU, i.e. it fails exactly when we need it. Set GEAK_GPU_REQUIRE_IDLE=0 to skip
# (e.g. deliberately co-tenanted screening runs). The default is 1 on BOTH paths: pool mode steps to
# another GPU when one is busy, single-GPU mode has nowhere to step and so fails loudly instead.
#
# The GPU id is mapped to its sysfs device through KFD topology, not render-minor arithmetic:
# render minors are consecutive on some hosts (MI210: renderD128..135) and spaced by eight on
# others, so `128 + 8 * id` silently probed nonexistent nodes. ROCr enumerates GPU agents in KFD
# node order (nodes with simd_count > 0), so physical index N is the N-th such node, and GPU_ID is
# logical within an inherited numeric ROCR_VISIBLE_DEVICES mask. Each node's location_id/domain
# must match the PCI address of its render node. GEAK_SYSFS_ROOT exists only for tests.
GEAK_SYSFS_ROOT="${GEAK_SYSFS_ROOT:-/sys}"
_gpu_sysfs_dev() {
    local id="$1" phys i=0 entry node props minor loc dom dev want
    case "$id" in ''|*[!0-9]*) echo "GPU id '$id' is not a non-negative integer" >&2; return 1 ;; esac
    phys="$id"
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        phys=""
        for entry in $(echo "$ROCR_VISIBLE_DEVICES" | tr ',' ' '); do
            case "$entry" in ''|*[!0-9]*)
                echo "ROCR_VISIBLE_DEVICES entry '$entry' is not a numeric index; cannot map GPU $id to sysfs" >&2
                return 1 ;;
            esac
            [ "$i" = "$id" ] && phys="$entry"
            i=$((i + 1))
        done
        [ -n "$phys" ] || { echo "GPU $id is outside ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES" >&2; return 1; }
    fi
    i=0
    for node in $(ls "$GEAK_SYSFS_ROOT/class/kfd/kfd/topology/nodes" 2>/dev/null | sort -n); do
        props="$GEAK_SYSFS_ROOT/class/kfd/kfd/topology/nodes/$node/properties"
        [ "$(awk '$1=="simd_count"{print $2}' "$props" 2>/dev/null)" -gt 0 ] 2>/dev/null || continue
        if [ "$i" = "$phys" ]; then
            minor="$(awk '$1=="drm_render_minor"{print $2}' "$props")"
            loc="$(awk '$1=="location_id"{print $2}' "$props")"
            dom="$(awk '$1=="domain"{print $2}' "$props")"
            case "$minor$loc${dom:-0}" in ''|*[!0-9]*)
                echo "KFD node $node has no numeric render minor/location" >&2; return 1 ;;
            esac
            dev="$(readlink -f "$GEAK_SYSFS_ROOT/class/drm/renderD${minor}/device" 2>/dev/null)" || dev=""
            want="$(printf '%04x:%02x:%02x.%x' "${dom:-0}" $(( (loc >> 8) & 255 )) $(( (loc >> 3) & 31 )) $(( loc & 7 )))"
            if [ -z "$dev" ] || [ "${dev##*/}" != "$want" ]; then
                echo "KFD node $node (PCI $want) does not match renderD${minor} (${dev:-missing})" >&2
                return 1
            fi
            echo "$dev"
            return 0
        fi
        i=$((i + 1))
    done
    echo "GPU $id (physical index $phys) has no KFD GPU node under $GEAK_SYSFS_ROOT" >&2
    return 1
}

# Returns 0 idle, 1 busy, 2 unknown. Unknown (unmappable id, unreadable telemetry) is never idle:
# callers fail loudly rather than measure on a GPU they could not inspect.
_gpu_is_idle() {
    local id="$1" dev busy vram
    dev="$(_gpu_sysfs_dev "$id")" || return 2
    busy="$(cat "$dev/gpu_busy_percent" 2>/dev/null)" || { echo "GPU $id: $dev/gpu_busy_percent unreadable" >&2; return 2; }
    vram="$(cat "$dev/mem_info_vram_used" 2>/dev/null)" || { echo "GPU $id: $dev/mem_info_vram_used unreadable" >&2; return 2; }
    case "$busy" in ''|*[!0-9]*) echo "GPU $id: non-numeric gpu_busy_percent '$busy'" >&2; return 2 ;; esac
    case "$vram" in ''|*[!0-9]*) echo "GPU $id: non-numeric mem_info_vram_used '$vram'" >&2; return 2 ;; esac
    [ "$busy" -le "${GEAK_GPU_MAX_BUSY_PCT:-5}" ] && [ "$(( vram / 1048576 ))" -le "${GEAK_GPU_MAX_VRAM_MB:-1024}" ]
}

# Lock files are named by the GPU's PCI address, not its id: an id is relative to the caller's
# ROCR_VISIBLE_DEVICES (a container limited to physical GPU 1 calls it GPU 0), so id-named locks let two
# runs on one GPU take different locks and two runs on different GPUs take the same one.
# A caller that opted out of inspection (GEAK_GPU_REQUIRE_IDLE=0) may lock an unmappable id, under a separate
# gpu_unmapped_<id> name that never aliases a PCI-named lock, and is told that ownership is not proven.
_gpu_lock_file() {
    local dev
    if dev="$(_gpu_sysfs_dev "$1")"; then
        echo "${LOCK_DIR}/gpu_${dev##*/}.lock"
    elif [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "0" ]; then
        echo "WARNING: GPU $1 has no PCI mapping; GEAK_GPU_REQUIRE_IDLE=0, so locking it by id only." >&2
        echo "${LOCK_DIR}/gpu_unmapped_${1}.lock"
    else
        echo "ERROR: cannot resolve GPU $1 to a PCI address for its lock (see above); refusing to run." >&2
        return 1
    fi
}

_unknown_idle_error() {
    echo "ERROR: cannot verify that GPU $1 is idle (see above); refusing to run on it." >&2
    echo "       Fix the device mapping, or set GEAK_GPU_REQUIRE_IDLE=0 only for deliberately co-tenanted runs." >&2
}

case "$GPU_SPEC" in
  *,*)
    # --- pool mode: block until some lane is free AND idle, then hold it for the whole command ---
    POOL="$(echo "$GPU_SPEC" | tr ',' ' ')"
    # When the wait started. Time spent blocked here is the SCHEDULER'S COST -- the price paid for
    # sharing GPUs instead of pinning one per engineer -- and it used to leave no trace at all: the
    # loop retries on `sleep 0.2` and only writes the use-log AFTER it wins a GPU, so an acquisition
    # that took ten minutes and one that took none were recorded identically.
    _wait_t0=$SECONDS
    _deadline=$(( _wait_t0 + ${GEAK_GPU_POOL_WAIT:-1200} ))
    GPU_ID=""
    while [ -z "$GPU_ID" ]; do
        for _g in $POOL; do
            # (1) flock: exclusive lane. Held until this process exits.
            _lock_file="$(_gpu_lock_file "$_g")" || exit 1
            exec {_fd}>"$_lock_file"
            if flock -n -x "$_fd"; then
                # We hold the lane. Only now check idleness -- checking before locking would race.
                if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ]; then
                    _idle_rc=0; _gpu_is_idle "$_g" || _idle_rc=$?
                    [ "$_idle_rc" = 2 ] && { _unknown_idle_error "$_g"; exit 1; }
                    if [ "$_idle_rc" != 0 ]; then
                        flock -u "$_fd"; exec {_fd}>&-   # foreign job on this GPU: try the next lane
                        continue
                    fi
                fi
                GPU_ID="$_g"; POOL_FD="$_fd"
                # Record which GPU of the pool was actually taken. Without this a pool acquisition
                # is unobservable after the fact: when a foreign tenant holds part of the pool the
                # loop above silently settles for a smaller set, and the run is still filed under
                # its original "N GPUs" label. One append-only line per acquisition makes the real
                # GPU set recoverable. Off unless GEAK_GPU_USE_LOG names a file.
                #
                # wait_s goes on the SAME line rather than into a new file: it is a property of this
                # acquisition, every reader already parses this line, and appending a field stays
                # backward-compatible with logs written before it existed. Nothing is added inside
                # the timed region -- the arithmetic runs after the GPU is already won.
                [ -n "${GEAK_GPU_USE_LOG:-}" ] && \
                    echo "{\"t\":$(date +%s),\"gpu\":$_g,\"pool\":\"$GPU_SPEC\",\"pid\":$$,\"mode\":\"pool\",\"wait_s\":$(( SECONDS - _wait_t0 ))}" \
                        >> "$GEAK_GPU_USE_LOG" 2>/dev/null
                break
            fi
            exec {_fd}>&-
        done
        if [ -z "$GPU_ID" ]; then
            [ "$SECONDS" -ge "$_deadline" ] && { echo "ERROR: no free+idle GPU in pool [$GPU_SPEC] after ${GEAK_GPU_POOL_WAIT:-1200}s" >&2; exit 1; }
            sleep 0.2
        fi
    done
    ;;
  *)
    # Single-GPU mode: unchanged from before this change. Falls through to the flock below.
    GPU_ID="$GPU_SPEC"
    ;;
esac

LOCK_FILE="$(_gpu_lock_file "$GPU_ID")" || exit 1

# (0) Reap ORPHANED hung rocm_agent_enumerator procs before running. aiter's import spawns one such
# subprocess per Python process for gfx detection; under GPU/KFD contention they HANG instead of
# exiting (<1s normally). With many parallel kernel jobs they pile up by the hundreds -> kernel
# task-count explosion -> whole-box hang (observed: 561 enumerators / 37k tasks on a swap=0 box).
# We kill ONLY ppid==1 (parent already dead) AND >60s old -> a live, in-use enumerator is never
# touched. Best-effort; must never fail the wrapper (set -e).
# OPT-IN (KERNEL_ENV_SKIP_ENUM_REAP=0): pgrep matches every enumerator on the host, and on a node
# shared with other jobs -- possibly under the same uid -- an orphan here is not proof that this
# run spawned it. GPU_ARCHS pinning in (3) already removes the spawn storm at its source.
if [ "${KERNEL_ENV_SKIP_ENUM_REAP:-1}" != "1" ]; then
    for _p in $(pgrep -f rocm_agent_enumerator 2>/dev/null || true); do
        _pp="$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        _et="$(ps -o etimes= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        if [ "${_pp:-0}" = "1" ] && [ -n "${_et:-}" ] && [ "${_et:-0}" -gt 60 ] 2>/dev/null; then
            kill -9 "$_p" 2>/dev/null || true
        fi
    done
fi

# (2) Per-workspace torch extension build cache (default: a hidden dir in the current workspace).
: "${TORCH_EXTENSIONS_DIR:=$PWD/.torch_ext}"
export TORCH_EXTENSIONS_DIR
mkdir -p "$TORCH_EXTENSIONS_DIR" 2>/dev/null || true

# (3) Compile for the selected GPU's arch only. HIP_VISIBLE_DEVICES does NOT
# filter rocminfo. When the launcher did not already provide a ROCR allocation,
# scope only this rocminfo subprocess to the locked physical GPU. Preserve an
# inherited ROCR mask because GPU_ID is then logical within that allocation.
# Refuse a genuinely mixed-ISA allocation: compiling for one ISA while locked
# to another is a silent wrong-arch result.
# Set KERNEL_ENV_KEEP_ARCH=1 to opt out (intentional multi-arch boxes).
_rocminfo_gpu_gfx_list() {
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || [ -z "${GPU_ID:-}" ]; then
        rocminfo 2>/dev/null
    else
        ROCR_VISIBLE_DEVICES="$GPU_ID" rocminfo 2>/dev/null
    fi | awk '
          /^ *Name: *gfx[0-9a-f]+/ && $2 != "gfx000" { print $2 }
        '
}

_pin_compile_arch() {
    [ "${KERNEL_ENV_KEEP_ARCH:-0}" = "1" ] && return 0
    local gfxs unique n
    gfxs="$(_rocminfo_gpu_gfx_list || true)"
    [ -z "${gfxs:-}" ] && return 0
    unique="$(printf '%s\n' $gfxs | sort -u)"
    n="$(printf '%s\n' $unique | grep -c . || true)"
    if [ "${n:-0}" -gt 1 ]; then
        echo "ERROR: gpu_lock.sh: mixed GPU ISAs in this pool ($unique); refusing to guess PYTORCH_ROCM_ARCH." >&2
        echo "       Run on a homogeneous pool or set KERNEL_ENV_KEEP_ARCH=1 with an explicit PYTORCH_ROCM_ARCH." >&2
        return 1
    fi
    _ARCH="$(printf '%s\n' $unique | head -1)"
    [ -n "${_ARCH:-}" ] && export PYTORCH_ROCM_ARCH="$_ARCH"
    # aiter's native detector spawns rocm_agent_enumerator per process; pinning
    # GPU_ARCHS avoids an enumerator storm as well as redundant multi-ISA builds.
    [ -n "${_ARCH:-}" ] && export GPU_ARCHS="${GPU_ARCHS:-$_ARCH}"
}

if [ -n "${POOL_FD:-}" ]; then
    # Pool mode: (1) this process ALREADY holds the lane exclusively (and verified it idle).
    # Re-locking the same file from the same process would be a no-op at best, so just run --
    # the lane stays held until we exit, which is what guarantees no two evaluations share a GPU.
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    _pin_compile_arch || exit 1   # (3) after the selected GPU is visible
    "$@"
else
    # Single-GPU mode BLOCKS TOO, and its wait must be measured for the same reason the pool's is.
    # Pinning does not mean "no contention": with 4 engineers over 2 pinned GPUs, two engineers share
    # each card and serialize on exactly this flock. That queueing is the pinned policy's own cost,
    # and comparing a measured pool wait against an assumed-zero pin wait would build the scheduler's
    # advantage into the instrument. Both paths measure, so the comparison is real.
    _wait_t0=$SECONDS
    (
        # (1) flock this GPU (exclusive for the duration of the command).
        flock -x -w 1200 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 1200s"; exit 1; }
        # Default 1, matching pool mode above. It was 0 here, so a PINNED engineer skipped the
        # foreign-work check entirely -- not "sampled it once", never ran it. That is how two
        # measurement cells ended up timed on a GPU another tenant was executing on. Both paths ask
        # the same question of the same driver; there is no reason for them to answer differently,
        # and the unsafe default was the one nobody had to opt into. The original rationale for 0
        # was that a single GPU has no alternative to step to, which is true -- but it argues for a
        # loud failure, not for measuring on a contaminated card. Set GEAK_GPU_REQUIRE_IDLE=0 to
        # restore the old behavior for deliberately co-tenanted runs.
        if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ]; then
            _idle_rc=0; _gpu_is_idle "$GPU_ID" || _idle_rc=$?
            [ "$_idle_rc" = 2 ] && { _unknown_idle_error "$GPU_ID"; exit 1; }
            if [ "$_idle_rc" != 0 ]; then
                echo "ERROR: GPU $GPU_ID has foreign work running (busy/VRAM above threshold); refusing to measure on it" >&2
                exit 1
            fi
        fi
        [ -n "${GEAK_GPU_USE_LOG:-}" ] && \
            echo "{\"t\":$(date +%s),\"gpu\":$GPU_ID,\"pool\":\"$GPU_SPEC\",\"pid\":$$,\"mode\":\"pin\",\"wait_s\":$(( SECONDS - _wait_t0 ))}" \
                >> "$GEAK_GPU_USE_LOG" 2>/dev/null
        export HIP_VISIBLE_DEVICES="$GPU_ID"
        _pin_compile_arch || exit 1   # (3) after the selected GPU is visible
        "$@"
    ) 200>"$LOCK_FILE"
fi
