# Selective robot vision data modeling

LeRobot demonstration data를 대상으로 영상 가시성과 행동 기록 신뢰도를 transition 단위로 평가하고, 선택적 가중치를 SmolVLA 학습에 적용하는 실험 코드입니다.

## 설치

```bash
pip install -e .
pip install -e ".[lerobot]"       # 실제 LeRobot 데이터와 SmolVLA 학습
pip install -e ".[segmentation]"  # prompted segmentation 모델
pip install -e ".[dev]"           # 개발 도구
```

## Cabinet 데이터셋 예시

기본 설정은 ALOHA mobile cabinet입니다. 전체 데이터 다운로드는 1.74 GB이므로 먼저 메타데이터를 확인하세요.

```bash
svdt inspect -c configs/dataset/cabinet.yaml
svdt download -c configs/dataset/cabinet.yaml --metadata-only
svdt download -c configs/dataset/cabinet.yaml --yes
```

가시성 분할은 기본적으로 비활성화되어 있습니다. 오프라인 마스크/트랙을 설정하거나 prompted 분할을 활성화하기 전까지 객체 가시성 항목은 `missing_evidence`로 표시됩니다.

## 주요 명령

```bash
svdt env
svdt score -c configs/dataset/cabinet.yaml -c configs/weighting/combined.yaml
svdt train -c configs/train/cabinet_smolvla.yaml --dry-run
```

합성 smoke 설정은 `configs/train/synthetic_smoke.yaml`에서 확인할 수 있습니다. 데이터셋별 사실과 실행 절차는 [docs/datasets.md](docs/datasets.md)를 참고하세요.

검토 라벨이 있는 경우 confidence workflow는 validation episode에서 원점수를 보정하고 held-out test에서 calibration을 평가합니다. 실행과 CSV 형식은 [docs/confidence.md](docs/confidence.md)를 참고하세요.
