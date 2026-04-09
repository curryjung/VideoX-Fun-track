# Track Builder UI Handoff

## 1) 작업 위치

- Repo: `VideoX-Fun`
- 프로젝트: `track_builder_ui`
- 프론트엔드 핵심 파일:
  - `frontend/src/App.tsx`
  - `frontend/src/components/TrackCanvas.tsx`
  - `frontend/src/components/TrackPreview.tsx`
  - `frontend/src/types.ts`

## 2) 현재 구현 상태 (완료된 기능)

- React + TypeScript 기반 단일 페이지 UI 동작
- 이미지 로드 2가지 지원
  - 버튼 업로드 (`Load Image`)
  - 캔버스 영역 drag-and-drop
- Point cloud 생성/편집
  - Rows/Cols 입력으로 square point cloud 생성 (`Apply Rows/Cols`)
  - 선택된 point cloud 전체를 그룹 단위로 드래그 이동
  - 시작 자세 수동 고정 (`Set Current As Start`)
- 시각화
  - 각 point를 서로 다른 색으로 렌더링 (가시성 개선)
  - 선택된 path 강조 렌더링
  - grid 표시 on/off + grid spacing 슬라이더
- point 간격 조절
  - `Point Spacing Scale` 슬라이더 변경 즉시 적용 (Apply 버튼 제거됨)
- 시간 기반 recording
  - 총 `FRAME_COUNT = 81`
  - 마우스 이동 거리와 무관하게, 마우스 홀드 중 120ms tick으로 frame 기록
  - Recording ON 상태에서만 tick 기록
  - 마지막 frame 도달 시 자동 `Recording OFF`
- 완료 후 미리보기
  - recording 완료 시 우측 `Track Preview On Image` 패널 표시
  - 프레임별 point 궤적 overlay 표시
- JSON import/export
  - 현재 path/keyframes/grid/image src를 문서로 저장/복원

## 3) 핵심 로직 요약

### A. Time-based recording

- `TrackCanvas.tsx`
  - `mousedown`으로 path 선택 후 드래그 시작 시 interval 시작
  - `setInterval(..., 120)`에서 `onRecordTick(pathId)` 호출
  - `mouseup`에서 interval 정리
- `App.tsx`
  - `handleRecordTick(pathId)`에서 `isRecording`이 true일 때만 frame 증가
  - `prevFrame -> nextFrame`으로 진행하며 keyframe 복제 저장
  - `nextFrame >= 80`이면 recording 자동 종료 및 완료 상태 세팅

### B. Group movement

- `handlePathDrag(pathId, dx, dy)`에서 개별 점이 아닌 현재 프레임의 전체 point set을 동일 벡터로 이동
- 이동값은 현재 frame keyframe에 반영

### C. Spacing scale

- `handlePointSpacingScaleChange(nextScale)`에서 centroid 기준 스케일링
- 선택 path의 모든 keyframe에 동일 비율 적용

## 4) 최근 사용자 요구와 반영 여부

- "point cloud를 그룹으로 이동" -> 반영 완료
- "rows/cols 변경" -> 반영 완료
- "point 간격 슬라이더 즉시 적용" -> 반영 완료
- "recording ON일 때만 기록" -> 반영 완료
- "프레임은 시간 기반으로만 진행" -> 반영 완료
- "Recording ON 버튼 빨간 강조" -> 반영 완료 (`recording-button recording` class)
- "81 frame 도달 시 자동 OFF" -> 반영 완료
- "완료 후 track preview on image" -> 반영 완료
- "point마다 다른 색" -> 반영 완료

## 5) 실행/검증

- 개발 서버
  - `cd track_builder_ui/frontend`
  - `npm run dev`
- 프로덕션 빌드
  - `npm run build`
  - 현재 기준 build 성공 확인됨

## 6) 다음 에이전트 우선 작업 제안

- UX polish
  - preview 항상 표시 + 완료 전에는 ghost/partial 표시 옵션 검토
  - path 다중 선택/전환 UX 개선
- 안정성
  - 파일 URL revoke 타이밍 정리 (`URL.createObjectURL` lifecycle 점검)
  - import JSON 스키마 validation 추가 (invalid data 방어)
- 기능 확장
  - 색상 preset/seed 고정 옵션
  - frame playback(재생/일시정지) 컨트롤
  - backend 연동 저장 API와 프론트 export/import 동기화
- 테스트
  - 핵심 로직 유닛 테스트 추가:
    - `ensureKeyframes`
    - `handleRecordTick` frame progression
    - spacing scale centroid 보존

## 7) 참고 메모

- 현재 구현은 "드래그 중 tick 기록" 패턴이다. Recording ON만으로 자동 재생/자동 기록은 하지 않는다.
- 미리보기 패널은 `isRecordingCompleted`가 true일 때만 렌더링된다.
- 선택 path 기준으로 편집이 이루어지므로, 다중 path 동시 recording은 아직 미지원.
