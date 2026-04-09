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

프론트 단독 실행도 가능하지만, API 저장/로드를 사용하려면 백엔드를 함께 실행하세요.

```bash
#install unicorn
sudo apt install -y unicorn
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API 문서: `http://127.0.0.1:8000/docs`

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
