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
mycobot qp --to 150 -60 200                    # QP 제어로 TCP 목표까지 이동 (qp extra 필요)
```

각 명령에 `--live`를 붙이면 실제 로봇을 사용합니다. `--read-only`를 쓰면 읽기만 하고 명령은 모두 버립니다.

### teleop 키

```
joint 모드         w e r t y u = +J1..+J6      s d f g h j = -J1..-J6
cartesian 모드     w/s ±X  a/d ±Y  r/f ±Z      u/j ±rx  i/k ±ry  o/l ±rz   (--ik qp 기본, --ik firmware)
gripper            , / . 한 단계씩            < / > 끝까지
control            space 정지(gate 닫기 + stop())   [ ] 스텝 1/2배, 2배   ? 도움말   q 종료
```

`--log`를 주면 키 입력마다 `recordings/teleop-*.csv`에 한 줄씩 기록합니다. `mycobot record`와 같은 `t_host` 시계를 쓰므로 두 파일을 시간으로 join할 수 있습니다.

## QP 제어 (`mycobot_control.qp`)

펌웨어 `send_coords` IK는 목표마다 감속해서 멈추기 때문에 짧은 목표를 이어 보내면 동작이 끊깁니다.
`qp/`는 매 tick(기본 20 Hz)마다 작은 QP를 풀어 다음 관절 명령을 계산하고, `command_joints`(`send_angles`)로 연속 스트리밍합니다.
관절 위치·속도·가속도 한계와 TCP 속도, 선택적인 바닥 평면이 페널티가 아니라 제약이므로 명령이 한계를 넘지 않고, 가속도 제약 덕분에 목표가 바뀌어도 속도가 튀지 않습니다.
펌웨어 IK는 대체 경로로 남아 있습니다(`teleop --ik firmware`).

```bash
uv sync --extra dev --extra qp        # numpy만 추가됩니다 (Jetson: pip install -e '.[arm,qp]', numpy 1.19에서 테스트)
mycobot qp --to 150 -60 200                          # dry-run: 계획 요약 + dry-run 세션으로 gate/한계 검사 재생
mycobot qp --to 150 -60 200 --rpy -180 0 -90 --duration 3
mycobot qp --joints 0 -30 -60 0 0 0 --live --speed 40
mycobot teleop --mode cartesian                      # 키 하나 = 짧은 min-jerk chunk를 QP로 실행
```

웹 콘솔 등에서 chunk를 실행할 때는 `ChunkExecutor`를 씁니다. 실제 시간은 호출하는 쪽이 관리하며, `tick()`을 `dt`마다 한 번 부릅니다.

```python
from mycobot_control.qp import ChunkExecutor, ControllerConfig, Kinematics, QPController

ex = ChunkExecutor(QPController(Kinematics(), ControllerConfig(dt=0.05)), q_measured, lookahead_ticks=2)
ex.set_cartesian_chunk(poses)          # 4x4 (mm), sample 0 = 지금, cfg.dt 간격. 진행 중에 바꿔도 멈추지 않음
ex.extend(more_poses)                  # 큐 끝에 이어 붙이기
while not ex.done:
    q_cmd = ex.tick(q_measured)        # measured가 12° 이상 벗어나면 TrackingError
    session.command_joints(q_cmd, measured=reading)
```

| 기본값 | 값 |
| --- | --- |
| `dt` | 0.05 s |
| 관절 속도 / 가속도 | 60 deg/s / 300 deg/s² (`mycobot qp`는 `--speed`에 맞춰 `0.96 × speed` deg/s, 최대 60) |
| TCP 속도 | 150 mm/s, 90 deg/s |
| 관절 한계 여유 | `JOINT_LIMIT_MARGIN_DEG` + 1 = 4° |
| J1 아끼기 | `joint_weights=(20, 1, 1, 1, 1, 1)` 또는 `--j1-weight 20` |

단위: 관절은 도, 위치는 mm입니다. 예외는 두 가지입니다. `Kinematics.jacobian()`의 열은 **라디안당**이고, `step()`의 `dq_prev`는 **tick당 도**입니다.
국소 미분 IK이므로 팔이 펴지는 특이점(J3 = 0)과 손목 특이점(J5 = ±90°)을 넘어 다른 IK 가지로 가지 못합니다. 큰 이동은 `goal_joints` 또는 planner(cuRobo)를 사용하세요.
아직 검증하지 않은 항목(`TODO(unverified)`)은 다음과 같습니다. speed 값과 deg/s의 대응(spec sheet의 100 = 약 120°/s를 가정), lookahead가 펌웨어의 정지-출발을 실제로 없애는지, Jetson에서 20 Hz로 읽기와 명령을 함께 처리할 수 있는지입니다. 바닥 제약은 TCP 한 점에만 적용됩니다.

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
