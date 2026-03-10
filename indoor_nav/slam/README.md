# Indoor SLAM Scaffold

This directory contains the Python-side SLAM integration boundary for the
indoor navigation stack.

Current contents:

- `types.py`: shared pose/tracking status dataclasses
- `base.py`: abstract client interface
- `orbslam3_client.py`: HTTP client for a local sidecar
- `protocol.md`: concrete sidecar contract
- `mock_sidecar.py`: mock implementation for client and preflight testing
- `calib/front_camera.yaml`: placeholder camera settings file

Current status:

- the repo can now configure, preflight, and talk to a SLAM sidecar
- the indoor agent can consume SLAM status in the control loop for:
  - motion gating
  - lost-tracking relocalization
  - pose-aware stuck recovery
- the only runnable sidecar in-tree is still the mock implementation
- the native ORB-SLAM3 sidecar scaffold lives under:
  - `external/orbslam3_sidecar`

Quick local test:

```bash
python -m indoor_nav.slam.mock_sidecar --host 127.0.0.1 --port 8765
python indoor_nav/check_indoor.py --goals indoor_nav/goals --policy heuristic --slam-backend orbslam3 --probe-model-load
```

Native sidecar bootstrap:

```bash
scripts/setup_orbslam3.sh
scripts/install_orbslam3_deps_ubuntu.sh   # Ubuntu helper, optional but practical
scripts/install_pangolin_from_source.sh   # if Pangolin is not packaged by apt
scripts/check_orbslam3_toolchain.sh
scripts/build_orbslam3_sidecar.sh
scripts/run_orbslam3_sidecar.sh --host 127.0.0.1 --port 8765
```

Important:

- `setup_orbslam3.sh` fetches the upstream ORB-SLAM3 tree into
  `external/ORB_SLAM3` and falls back to a GitHub tarball if `git clone`
  stalls or fails.
- `install_orbslam3_deps_ubuntu.sh` installs the baseline Ubuntu-native build
  dependencies and handles the common `libpangolin-dev` package split.
- `install_pangolin_from_source.sh` installs Pangolin from source when apt
  does not package the development files. By default it installs into
  `external/pangolin-install`, so it can work without root.
- `check_orbslam3_toolchain.sh` validates the actual CMake dependency path
  (`find_package(OpenCV)`, `find_package(Pangolin)`) before the sidecar build
  runs.
- `build_orbslam3_sidecar.sh` only builds the current native scaffold.
- the scaffolded native binary is still a stub until the real ORB-SLAM3 HTTP
  server implementation replaces it.
