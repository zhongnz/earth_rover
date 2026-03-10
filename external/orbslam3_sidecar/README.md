# ORB-SLAM3 Sidecar Scaffold

This directory is the native sidecar target for the indoor SLAM integration.

Current state:
- Python-side client, protocol, preflight, and agent integration already exist.
- The real ORB-SLAM3 backend is not vendored here yet.
- The runnable path in-repo is still the mock sidecar at `indoor_nav/slam/mock_sidecar.py`.

The intended build shape is:

```text
external/ORB_SLAM3/           # upstream ORB-SLAM3 checkout
external/orbslam3_sidecar/    # thin HTTP wrapper around ORB-SLAM3
```

The sidecar is expected to implement the HTTP contract in:

```text
indoor_nav/slam/protocol.md
```

## Prerequisites

1. Native Linux machine, preferably x86_64
2. ORB-SLAM3 source checkout at `external/ORB_SLAM3`
3. Built ORB-SLAM3 dependencies (Pangolin, OpenCV, Eigen, etc.)
4. Real front-camera calibration file

## Setup

Clone ORB-SLAM3 into the expected path:

```bash
scripts/setup_orbslam3.sh
scripts/install_orbslam3_deps_ubuntu.sh   # Ubuntu helper, if needed
scripts/install_pangolin_from_source.sh   # if libpangolin-dev is unavailable
```

By default, `install_pangolin_from_source.sh` installs Pangolin into the repo at
`external/pangolin-install`, so it can be used without root access. The build
and toolchain scripts automatically honor that prefix.

## Build

```bash
scripts/check_orbslam3_toolchain.sh
scripts/build_orbslam3_sidecar.sh
```

## Run

```bash
scripts/run_orbslam3_sidecar.sh --host 127.0.0.1 --port 8765
```

## Notes

- This scaffold is intentionally thin. The Python agent should keep ownership of
  control-loop orchestration and safety.
- The sidecar should only own SLAM state, tracking, map lifecycle, and pose
  estimation.
- If ORB-SLAM3 is not available, keep using the mock sidecar for client and
  preflight development.
