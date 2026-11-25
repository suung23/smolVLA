# Dataset configuration

## ALOHA mobile cabinet

설정 파일: `configs/dataset/cabinet.yaml`

- LeRobot 저장소: `lerobot/aloha_mobile_cabinet`
- 라이선스: Apache-2.0
- 기록 주파수: 50 fps
- 카메라: high, left wrist, right wrist
- state/action: 각각 14차원 joint 벡터
- task: 상단 캐비닛을 열고 pot을 넣은 다음 닫기
- 동작 의미: 14개 관절의 절대 위치 목표. gripper action 인덱스는 6, 13.

전체 데이터 다운로드 전에 메타데이터만 확인할 수 있습니다.

```bash
svdt inspect -c configs/dataset/cabinet.yaml
svdt download -c configs/dataset/cabinet.yaml --metadata-only
svdt download -c configs/dataset/cabinet.yaml --yes
```

캐비닛 손잡이, 문, pot을 색상만으로 안정적으로 분할할 수 없어 기본 설정은 segmentation을 비활성화합니다. 오프라인 mask/track을 `mask_track`으로 연결하거나 prompted backend를 설정할 수 있습니다. 유효한 분할 evidence가 없으면 객체 관련 값은 `missing_evidence`로 기록됩니다.

이 데이터셋에 대응하는 검증된 simulator가 등록되어 있지 않으므로 rollout 성공률은 `not_available`입니다. Offline action metrics를 사용하세요.
