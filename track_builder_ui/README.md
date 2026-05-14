# Track Builder UI

이미지 위에서 point cloud track을 만들고, 시간 흐름(프레임) 기준으로 motion을 기록하는 UI입니다.  
Frontend는 React + TypeScript(Vite), Backend는 FastAPI로 구성되어 있습니다.

## 프로젝트 구조

- `frontend/`: 트랙 편집 UI
- `backend/`: 저장/로드용 FastAPI 서버
- `data/`: 로컬 데이터 저장 경로(백엔드 사용 시)
- `HANDOFF_TRACK_BUILDER_UI.md`: 현재 구현 상태 인수인계 문서

## 요구 사항

- Node.js + npm
- Python 3.10+ (권장)

## 1) Frontend 실행

```bash

# install nodejs and npm
sudo apt update
sudo apt install -y nodejs npm

cd frontend
npm install
npm run dev
```

- 접속 주소: `http://127.0.0.1:5173`
- 첫 화면에 데모 배경이 뜨며, `Load Image` 또는 drag-and-drop으로 이미지 교체 가능

### Frontend 빌드 확인 (선택)

```bash
cd frontend
npm run build
```

## 2) Backend 실행 (선택)

프론트 단독 실행도 가능하지만, 서버 export, queue/archive, video generation을 사용하려면 백엔드를 함께 실행하세요.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- API 문서: `http://127.0.0.1:8000/docs`

## 3) Generation 실행 모드

### Simple mode: backend가 job마다 추론 subprocess 실행

가장 단순한 모드입니다. Frontend + Backend만 실행하면 됩니다. 단, job마다 모델을 다시 GPU에 로드합니다.

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun/track_builder_ui/backend
source .venv/bin/activate

TRACK_BUILDER_PYTHON_BIN=/usr/local/bin/python \
TRACK_BUILDER_CUDA_VISIBLE_DEVICES=6 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Persistent worker mode: 모델을 GPU에 계속 유지

Backend는 job 생성/조회만 담당하고, 별도 worker가 queue를 처리합니다. 생성 작업을 많이 할 때 권장합니다.

터미널 1 - Backend:

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun/track_builder_ui/backend
source .venv/bin/activate

TRACK_BUILDER_RUNNER_MODE=external \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

터미널 2 - Persistent worker:

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun

CUDA_VISIBLE_DEVICES=6 \
python examples/wan2.1_fun_track/run_track_i2v_worker_experimental.py
```

터미널 3 - Frontend:

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun/track_builder_ui/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

> `run_track_i2v_worker_experimental.py`는 기존 `predict_i2v_track.py`를 수정하지 않고 helper를 import해서 사용하는 실험용 persistent worker입니다.

### Wan-Move mode로 실행

기본 generation backend는 `track_head` mode를 사용합니다. Wan-Move로 학습된 checkpoint를 쓰려면 `TRACK_BUILDER_TRACK_CONDITION_MODE=wan_move`를 지정하세요.

#### Simple mode

Backend subprocess가 직접 Wan-Move inference를 실행합니다.

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun/track_builder_ui/backend
source .venv/bin/activate

TRACK_BUILDER_TRACK_CONDITION_MODE=wan_move \
TRACK_BUILDER_CUDA_VISIBLE_DEVICES=6 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Persistent worker mode

Backend와 worker 양쪽에 같은 Wan-Move 설정을 넣어주세요. Frontend의 `4) Generation` 섹션은 backend의 `/api/runner/config` 값을 표시하므로, backend env와 worker env가 다르면 화면 표시와 실제 실행 모델이 달라질 수 있습니다.

터미널 1 - Backend:

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun/track_builder_ui/backend
source .venv/bin/activate

TRACK_BUILDER_RUNNER_MODE=external \
TRACK_BUILDER_TRACK_CONDITION_MODE=wan_move \
TRACK_BUILDER_CUDA_VISIBLE_DEVICES=6 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

터미널 2 - Persistent worker:

```bash
cd /data/project-vilab/jaeseok/VideoX-Fun
source .venv-videoxfun/bin/activate
TRACK_BUILDER_TRACK_CONDITION_MODE=wan_move \
CUDA_VISIBLE_DEVICES=6 \
python examples/wan2.1_fun_track/run_track_i2v_worker_experimental.py
```

Wan-Move 기본값은 `evaluation/davis_track_eval_wan_move.sh`와 맞춰져 있습니다.

- 기본 checkpoint: `checkpoints/wan_track_wan_move_condition_bin8_train_78k_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-11800`
- 기본 `TRACK_BUILDER_WAN_MOVE_TEMPORAL_STRIDE`: `0` (`<=0`이면 VAE temporal compression ratio를 자동 사용)
- 기본 track sampling: `TRACK_BUILDER_TRACK_MAX_POINTS=1500`, `TRACK_BUILDER_TRACK_POINT_SAMPLE_MODE=random`, `TRACK_BUILDER_TRACK_SORT_SELECTED_INDICES=true`, `TRACK_BUILDER_TRACK_POINT_ID_MODE=original`

checkpoint를 직접 지정하려면 mode별 exp/ckpt env 또는 전체 경로 env를 사용하세요.

```bash
# mode별 exp/ckpt 지정
TRACK_BUILDER_TRACK_CONDITION_MODE=wan_move \
TRACK_BUILDER_WAN_MOVE_EXP_NAME=wan_track_wan_move_condition_bin8_train_78k_dropout_first-frame_0p1_text_0p1_track_0p1 \
TRACK_BUILDER_WAN_MOVE_CKPT=11800 \
...

# 또는 checkpoint 디렉터리/파일 경로 직접 지정
TRACK_BUILDER_TRACK_CONDITION_MODE=wan_move \
TRACK_BUILDER_TRANSFORMER_CHECKPOINT_PATH=/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/.../checkpoint-11800 \
...
```

### Generation 환경변수

- `TRACK_BUILDER_CUDA_VISIBLE_DEVICES`: simple mode에서 backend subprocess가 사용할 GPU index. 기본값은 `6`.
- `CUDA_VISIBLE_DEVICES`: persistent worker가 사용할 GPU index.
- `TRACK_BUILDER_PYTHON_BIN`: simple mode에서 추론 subprocess에 사용할 Python.
- `TRACK_BUILDER_TRACK_CONDITION_MODE`: track conditioning backend. `track_head` 또는 `wan_move`. 기본값은 `track_head`.
- `TRACK_BUILDER_TRANSFORMER_CHECKPOINT_PATH`: 사용할 checkpoint 경로를 직접 지정.
- `TRACK_BUILDER_EXP_NAME`, `TRACK_BUILDER_CKPT`: mode와 무관하게 checkpoint exp/ckpt를 override.
- `TRACK_BUILDER_TRACK_HEAD_EXP_NAME`, `TRACK_BUILDER_TRACK_HEAD_CKPT`: `track_head` mode 전용 checkpoint exp/ckpt override.
- `TRACK_BUILDER_WAN_MOVE_EXP_NAME`, `TRACK_BUILDER_WAN_MOVE_CKPT`: `wan_move` mode 전용 checkpoint exp/ckpt override.
- `TRACK_BUILDER_WAN_MOVE_TEMPORAL_STRIDE`: Wan-Move track frame을 latent frame에 매핑할 temporal stride. 기본값은 `0`이며 자동 설정.
- `TRACK_BUILDER_TRACK_MAX_POINTS`: inference에 사용할 track point 수. 기본값은 `track_head=2000`, `wan_move=1500`.
- `TRACK_BUILDER_TRACK_POINT_SAMPLE_MODE`: point sampling 방식. `random` 또는 `uniform`. 기본값은 `random`.
- `TRACK_BUILDER_TRACK_SORT_SELECTED_INDICES`: random sampling 후 원래 index 순서 정렬 여부. 기본값은 `track_head=false`, `wan_move=true`.
- `TRACK_BUILDER_TRACK_POINT_ID_MODE`: point id 부여 방식. 기본값은 `track_head=local`, `wan_move=original`.
- `TRACK_BUILDER_JOBS_ROOT`: job/archive 저장 루트. 기본값은 `asset/track_builder_jobs`.

## 빠른 사용 순서

1. `Apply Rows/Cols`로 point cloud 생성
2. 필요 시 `Set Current As Start`로 시작 자세 지정
3. `Recording ON`으로 전환
4. 마우스 왼쪽 버튼을 누른 채 point cloud를 드래그하면 120ms tick 기준으로 기록 진행
5. 81프레임 도달 시 자동으로 Recording OFF
6. 완료되면 우측 `Track Preview On Image`에서 궤적 확인

## Florence 이미지 캡셔닝

- 좌측 `1) Input` 섹션에서 `Florence Task Prompt`를 설정하고 `Generate Caption (Florence)` 버튼을 눌러 캡션 생성 가능
- 기본 프롬프트는 `<MORE_DETAILED_CAPTION>`이며, 필요 시 `<OD>` 같은 Florence 태스크 토큰으로 변경 가능
- 프론트엔드는 백엔드 `POST /api/images/caption`을 호출함
- 기본은 Vite 프록시(`/api -> http://127.0.0.1:8000`)를 사용하며, 필요 시 `VITE_BACKEND_URL`로 백엔드 주소를 직접 지정 가능
- `Export Track Package` 시 ZIP 내부 `processed_832x480_fps16/image_caption.txt`로 캡션 텍스트가 함께 저장됨

## 문제 해결

- 포트 충돌 시:
  - frontend: `npm run dev -- --port 5174`
  - backend: `uvicorn app.main:app --reload --port 8001`
- 의존성 꼬임 시:
  - frontend: `rm -rf node_modules package-lock.json && npm install`
  - backend: 가상환경 삭제 후 재생성
- `transformers` import에서 `huggingface-hub` 버전 에러가 나면 generation에 사용하는 Python 환경의 버전을 맞추세요:
  - `python -m pip install "huggingface-hub>=0.26.0,<1.0"`
