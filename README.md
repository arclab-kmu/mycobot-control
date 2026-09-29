# mycobot-control

UROP에서 사용하는 myCobot 280 (JetsonNano, adaptive gripper)을 조작하는 프로젝트입니다.
`~/manipulation/nero-control`(AgileX NERO)과 같은 구조를 따릅니다. 핵심 원칙은 세 가지입니다.

- 기본은 **dry-run**입니다. `--live`를 주고 `yes`를 입력해야만 실제 로봇에 명령이 나갑니다.
- 모든 명령은 control gate를 통과해야 합니다. deadman(0.5초 입력 없으면 gate 닫힘)과 fault latch가 함께 동작합니다.
- ROS 2와 pymycobot은 선택 사항입니다. 기본 import 경로에는 둘 다 없습니다.

## 설치

```bash
uv sync --extra dev               # 개발/dry-run 전용 (pymycobot 없음)
uv sync --extra dev --extra arm   # 실제 로봇용 (pymycobot 포함)
uv run pytest                     # 하드웨어 없이 전체 테스트
```

주의: `uv sync --extra dev`만 실행하면 이미 설치된 pymycobot이 **삭제**됩니다.
Jetson에서 uv를 쓸 수 없다면 `pip install -e '.[arm]'`를 사용하세요. Python 3.8 이상이 필요합니다.

## 연결 방식

| 실행 위치 | 옵션 | 비고 |
| --- | --- | --- |
| Jetson 본체 | `--transport serial` (기본값), `/dev/ttyTHS1` @ 1000000 | vendor JN 스크립트와 같은 설정입니다. 사용자가 `dialout` 그룹에 있어야 합니다 |
| PC | `--transport socket --host <jetson-ip>` (포트 9000) | Jetson에서 pymycobot의 socket server(GitHub `demo/`)를 먼저 실행해야 합니다. 아직 검증하지 않았습니다 |

serial 모드에서는 vendor ROS 노드와 같은 `/tmp/mycobot_lock`을 세션 내내 잡고 있습니다. 따라서 ROS 노드와 동시에 실행할 수 없습니다.

## 명령어

uv 환경에서는 앞에 `uv run`을 붙입니다(예: `uv run mycobot info`). pip으로 설치했다면 그대로 실행합니다.

```bash
mycobot info                                   # 상태 보고 (live에서도 항상 read-only)
mycobot teleop                                 # 키보드 조인트 teleop (dry-run)
mycobot teleop --mode cartesian                # send_coords로 좌표 teleop (펌웨어 IK)
mycobot joints 0 -30 -60 0 0 0                 # 조인트 목표(도)로 waypoint 단위 이동
mycobot record --duration 10 --gripper         # CSV + .meta.json 기록 (명령 불가)
```

각 명령에 `--live`를 붙이면 실제 로봇을 사용합니다. `--read-only`를 쓰면 읽기만 하고 명령은 모두 버립니다.

### teleop 키

```
joint 모드         w e r t y u = +J1..+J6      s d f g h j = -J1..-J6
cartesian 모드     w/s ±X  a/d ±Y  r/f ±Z      u/j ±rx  i/k ±ry  o/l ±rz
gripper            , / . 한 단계씩            < / > 끝까지
control            space 정지(gate 닫기 + stop())   [ ] 스텝 1/2배, 2배   ? 도움말   q 종료
```

`--log`를 주면 키 입력마다 `recordings/teleop-*.csv`에 한 줄씩 기록합니다. `mycobot record`와 같은 `t_host` 시계를 쓰므로 두 파일을 시간으로 join할 수 있습니다.

## 첫 하드웨어 bring-up 순서

아래 순서를 건너뛰지 마세요. 각 단계에서 나온 값은 `safety/limits.py`의 `TODO(unverified)` 항목에 날짜와 함께 기록합니다.

1. `mycobot info --live`로 연결, 펌웨어 버전, 에러 코드, 컨트롤러 조인트 한계가 표와 일치하는지 확인합니다.
2. `mycobot record --live --duration 10`으로 실제 달성 Hz와 read latency를 측정합니다.
3. `mycobot teleop --live --joint-step 1`로 키를 한 번에 하나씩 누르며 각 조인트의 부호와 방향을 확인합니다.
4. gripper 키 `.`를 한 번 눌러 값이 커질 때 열리는지 확인합니다. 아직 검증하지 않았습니다.
5. `--mode cartesian`의 축 방향을 확인합니다. 펌웨어 base frame은 URDF와 z축으로 90° 돌아가 있다고 vendor가 명시하고 있습니다.

전원을 차단할 수 있는 방법을 항상 손이 닿는 곳에 두세요. software gate는 안전 장치가 아닙니다.
`release_all_servos()`나 `power_off()`는 자동으로 호출하지 않습니다. 둘 다 토크가 빠져서 들린 팔이 떨어집니다.

## 관련 저장소

- `~/manipulation/mycobot_ros2`: vendor ROS 2 패키지와 URDF (`mycobot_description/urdf/mycobot_280_jn/`)
- `~/manipulation/arc-mycobot`: Isaac Lab reach/lift 시뮬레이션. 녹화 CSV의 `j*_rad` 열과 joint 순서가 이 저장소의 URDF 순서와 같습니다.
