# 데이터 출처 (자동 생성: scripts/fetch_data.py)

커밋 SHA와 sha256으로 고정된 스냅샷입니다. 라이선스가 재배포를 허용하는 세트만 이 폴더에 복사했고, 라이선스가 없는 세트는 런타임에 `data/cache/`로만 내려받습니다.

| 세트 | 저장소 | 커밋 | 파일 | sha256 | 라이선스 | 재배포 | 행 수 |
|---|---|---|---|---|---|---|---|
| themsquared | https://github.com/themsquared/jev-calibration | `311daf22fdae` | `tasks.jsonl` | `a10a7ed68c6dd063…` | Apache-2.0 | 예 | 240 |
| themsquared | https://github.com/themsquared/jev-calibration | `311daf22fdae` | `LICENSE` | `cfc7749b96f63bd3…` | Apache-2.0 | 예 |  |
| godspede | https://github.com/godspede/construct-auto-classifier | `881fb45a96af` | `bench/battery.jsonl` | `56e57aebf1082826…` | Apache-2.0 | 예 | 181 |
| godspede | https://github.com/godspede/construct-auto-classifier | `881fb45a96af` | `bench/holdout.jsonl` | `284c2f7b7d6c6342…` | Apache-2.0 | 예 | 79 |
| godspede | https://github.com/godspede/construct-auto-classifier | `881fb45a96af` | `bench/real-cases.jsonl` | `38ee9706fe9d7bc8…` | Apache-2.0 | 예 | 151 |
| godspede | https://github.com/godspede/construct-auto-classifier | `881fb45a96af` | `bench/world.json` | `ee7f2005019ccf39…` | Apache-2.0 | 예 |  |
| godspede | https://github.com/godspede/construct-auto-classifier | `881fb45a96af` | `LICENSE` | `c71d239df91726fc…` | Apache-2.0 | 예 |  |
| laya_guard | https://github.com/scs0209/laya-guard | `3859853dd693` | `eval/testset_shell.jsonl` | `ede1c6f9a0cb00cc…` | MIT | 예 | 71 |
| laya_guard | https://github.com/scs0209/laya-guard | `3859853dd693` | `eval/testset_ko_30.jsonl` | `485cc47c5e178cac…` | MIT | 예 | 30 |
| laya_guard | https://github.com/scs0209/laya-guard | `3859853dd693` | `eval/testset_incidents.jsonl` | `8f5fadc632bca4da…` | MIT | 예 | 17 |
| laya_guard | https://github.com/scs0209/laya-guard | `3859853dd693` | `LICENSE` | `8fa0a3b302e2d203…` | MIT | 예 |  |
| aitejiu | https://github.com/Aitejiu/jev-harness-lab | `29ddf8210e3a` | `eval/datasets/bash_commands.jsonl` | `81c610d3a6a9f075…` | 없음(재배포 불가, 런타임 다운로드) | 아니오(런타임) | 130 |

`laya_guard/testset_incidents.jsonl`은 원본 레코드 2건의 `id`, `incident` 값만 `scripts/fetch_data.py`의 `VENDOR_EDITS`대로 바꿔 동봉했습니다. 명령과 라벨은 원본과 같습니다. 위 표의 sha256은 원본 파일 기준이고, 동봉본의 sha256은 `2bf062dc9d775946…`입니다.

라벨 매핑과 제외 규칙은 `src/dalab/gate/datasets.py`에 있습니다.
