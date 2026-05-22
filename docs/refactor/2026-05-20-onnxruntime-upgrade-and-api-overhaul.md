# easy-onnx — ONNX Runtime 업그레이드 및 전체 리팩토링 계획

## Context

`easy-onnx`는 ONNX Runtime 1.18.1을 vendored로 번들하여 C++17로 래핑한 추론 라이브러리(`aiRunner` 공유 라이브러리)다. 단일 모델 추론(전역 상태), 배치, async, 멀티 모델 인스턴스, 로깅, macOS arm64 + Windows x64 빌드까지 갖춰진 v0.1.0 상태. 그러나 다음 부채가 누적되어 있다.

- **공개 C ABI 불일치**: 단일 모델 API는 `void` 반환 + 전역 상태(`InitModel`, `RunModelFloat`, `GetFlattenedOutput`), 멀티 모델 API는 `bool` 반환 + 핸들 기반(`CreateModelInstance`, `RunModelInstanceFloat`). 같은 동작이 두 가지 형태로 공존해서 사용자가 어느 쪽을 써야 할지 혼란.
- **이중 상태**: `aiRunner.cpp`의 전역 정적(`workflow`, `flattened_output`, `instance_outputs`, `user_log_callback`)과 `ModelManager` 싱글톤이 같은 데이터를 중복 보관. 스레드 안전성 보장 안 됨(특히 async 결과 캐시 업데이트, `aiRunner.cpp:215-216, 242-243`).
- **로깅 우회**: Logger 모듈이 있는데도 `Model.cpp`/`Workflow.cpp`에 `std::cout`/`std::cerr` 직접 사용 약 58군데(로그 레벨, 콜백, 파일 출력 모두 무시됨).
- **ORT 1.18.1 정체**: 약 1년 이상 묵힘. CoreML EP의 MLProgram 모드, DirectML 최신 디바이스 지원, 성능 개선이 1.20+에서 반영. CoreML/DML factory 시그니처가 안정적이므로 업그레이드 비용 낮음.

**목표**: ORT 최신 stable(1.26.0)로 올리면서 누적 부채를 정리하고, 공개 C ABI를 **핸들 기반 단일 형태**로 일관화한 v0.2.0을 배포한다. 메이저 breaking change이므로 외부 소비자(현 시점엔 문서상 C++ 예제와 Unity P/Invoke 가이드)는 마이그레이션 가이드와 함께 제공.

## Decisions (with user)

| 항목 | 결정 |
|---|---|
| 범위 | **ORT 업그레이드 + 내부 부채 정리 + 공개 API 일관화(Breaking)**. Linux 추가, 신규 EP(CUDA/TensorRT), CI는 본 스코프 밖. |
| API 형태 | **핸들 기반으로 통일**. 전역 상태 단일 모델 API는 전부 제거. 모든 호출은 `instance_id`(또는 동등한 핸들)를 받음. 모든 함수 `bool` 반환. |
| ORT 타겟 | **1.26.0** (2026-05-08 릴리스, GitHub release 최신 stable). 1.21.x로 잡았다가 실제 확인 결과 1.26.0이 최신이라 상향 조정. |
| Windows DirectML EP | **Step 1 범위에서 제외 (deferred)**. upstream ORT 1.25부터 DirectML이 GitHub release 표준 패키지에서 빠지고, `Microsoft.ML.OnnxRuntime.DirectML` NuGet도 1.24.4가 마지막. Windows 가속 EP 전략(1.24.4 pin / DML 제거 후 CPU만 / WebGPU EP로 대체) 결정은 별도 트랙으로 분리. |
| 진행 방식 | **4단계 phased**. 각 단계 별도 PR/머지 가능 단위로 분리. |
| 네이밍 컨벤션(권장) | C ABI는 현재 다수파인 `PascalCase`(예: `CreateModelInstance`, `RunModelFloat`) 유지. 내부 C++ 헬퍼는 `snake_case`(현 `init_model`, `setSessionOption` 등 일부 camelCase는 snake_case로 정렬). 본 계획의 Step 3에서 한 번에 정렬. |

---

## Phased Plan

### Step 1 — ORT 1.18.1 → 1.26.0 업그레이드 (macOS 한정, 기존 API 그대로)

**상태**: 2026-05-20 완료.

**의도**: macOS 바이너리 교체 + 빌드 그린 + 단일 추론 회귀 없음 확인까지가 한 PR.

수행한 작업:
1. `Library/ONNXRuntime/mac/`을 ORT 1.26.0 (`onnxruntime-osx-arm64-1.26.0.tgz`)으로 일괄 교체. 1.26.0 추가 헤더(`onnxruntime_env_config_keys.h`, `onnxruntime_ep_c_api.h`, `onnxruntime_ep_device_ep_metadata_keys.h`, `provider_options.h`, `webgpu_provider_factory.h`, `core/providers/`)와 `lib/cmake/`, `lib/pkgconfig/` 포함. 디버그 심볼(`.dSYM`)은 용량 사유로 제외.
2. `Library/ONNXRuntime/windows/`는 **변경 없음** — DirectML EP 전략 결정 후 별도 PR. 본 PR에서는 Windows 빌드 동작은 1.18.1 그대로 유지(이 리포에서 Windows 빌드를 돌리지 않으므로 회귀 없음).
3. dylib symlink 체계 갱신: 1.26.0의 `install_name`이 `@rpath/libonnxruntime.1.dylib`(major-version symlink 기반)로 바뀌어, post-build 스테이징 단계에서 **`libonnxruntime.1.dylib` symlink를 추가 생성**해야 함. `CppLibrary/CMakeLists.txt`와 `CppApplication/CMakeLists.txt` 양쪽의 POST_BUILD/install 블록에 반영.
4. CMakeLists.txt의 `libonnxruntime.1.18.1.dylib` 문자열 8군데 → `libonnxruntime.1.26.0.dylib` 일괄 치환 (`CppLibrary/CMakeLists.txt`, `CppApplication/CMakeLists.txt`).
5. **ORT 1.26 동작 변경에 대응**: `Model::setSessionOption(cpu_use=false)`에서 `TryAppendAcceleratedEP`를 호출하기 전에 `Ort::Env`를 lazy-init 하도록 수정 (`CppLibrary/src/Model.cpp`). 1.26부터 EP factory는 default logger를 요구하는데 그 logger는 `Ort::Env` 생성 시 등록되므로, 기존 "setSessionOption → setModel" 순서로는 `Attempt to use DefaultLogger but none has been registered` 예외가 발생함. 같은 이유로 `setModel`도 env가 이미 있으면 재사용하도록 조건부 생성.
6. 회귀 검증: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j` 성공. `./build/CppApplication/CppApplication /tmp/mnist.onnx`(CPU)와 `--gpu`(CoreML) 모두 단일 추론 통과(로그 `[INFO ] EP appended: CoreML` 확인). CoreML 경로의 첫 element 출력값이 CPU 경로와 동일(-57.3601). `otool -L`로 staged dylib들이 `@rpath/libonnxruntime.1.dylib`을 통해 정상 resolve됨을 확인.

**완료 기준** (실제 결과):
- ✅ Build green (Release, -j).
- ✅ Single-sample inference 통과 (CPU + CoreML 둘 다).
- ⚠️ CppApplication batch/async/multi-instance 시나리오(2~7)는 fixed-batch=1인 MNIST/Squeezenet 모델 제약으로 종단 검증 미실시. 이는 1.18.1에서도 동일하게 실패하던 모델 한정 제약이므로 회귀 아님. 동적 배치 모델을 test fixture로 추가하는 것은 Step 2/3 후속 작업.

수정/추가 파일:
- `Library/ONNXRuntime/mac/{include,lib,GIT_COMMIT_ID,LICENSE,Privacy.md,README.md,ThirdPartyNotices.txt,VERSION_NUMBER}` (전체 교체)
- `CppLibrary/CMakeLists.txt` (라인 75–148)
- `CppApplication/CMakeLists.txt` (라인 60–84)
- `CppLibrary/src/Model.cpp` (`setSessionOption`, `setModel`의 env lazy-init)

### Step 2 — 내부 부채 정리 (공개 API 표면 유지) ✅ 완료 (2026-05-20)

**의도**: 사용자 가시 API는 그대로 두고 내부만 정리. Step 3에서 API를 갈아엎기 전에 안전망 마련.

**적용 내역**:
1. **전역 정적 상태 제거** — `aiRunner.cpp`의 9개 file-scope static을 모두 제거:
   - 단일 모델 상태 (`workflow`, `flattened_output`, `original_shape`, `async_future`, `async_result`) → `ModelManager`의 **default-instance** 상태(`default_id_`, `default_async_future_`, `default_async_result_`)로 이동. 별도 `instance_id` 예약 없이 `instances_` 맵을 그대로 공유하며 ID만 추적.
   - 멀티 모델 캐시 (`instance_outputs`, `instance_shapes`) → 제거. 사본 보관은 무의미했음(워크플로우 vector가 동일 생명주기). `getFlattenedOutput(id)`이 workflow의 vector 참조를 직접 반환.
   - 로그 콜백 브리지 (`user_log_callback`, `user_log_data`) → `Logger`의 `c_callback_`/`c_callback_user_data_` 멤버로 이동, `Logger::setCCallback`/`clearCCallback` API 추가.
2. **로깅 일원화** — `Model.cpp`, `Workflow.cpp`, `DataLoader.cpp`, `ModelManager.cpp`의 모든 `std::cout`/`std::cerr` 호출(약 30+ 군데)을 `Logger::info()/debug()/warn()/error()`로 치환. 디버그 수준의 텐서 shape 트레이스는 `getLogLevel() <= DEBUG` 가드로 비용 절감. `iostream` 직접 사용은 이제 `Logger.cpp`(콘솔 sink)와 pch.h에 한정.
3. **DirectML device_id 노출 준비** — `TryAppendAcceleratedEP(Ort::SessionOptions&, int device_id = 0)`로 시그니처 확장. macOS CoreML은 device_id 무시, DML은 디바이스 선택에 사용.
4. **소유권 명세** — Workflow 라이프사이클은 `ModelManager::instances_` 단독. `aiRunner.cpp`에 별도 `shared_ptr<Workflow>` 잔재 없음.
5. **동시성 보호** — `ModelManager`에 `default_mutex_` 추가하여 `default_id_`/`default_async_future_`/`default_async_result_` 접근 직렬화. 기존 `instances_mutex_`는 인스턴스 맵 보호 그대로. 잠금 순서는 `default_mutex_ → instances_mutex_` 단방향(주석 명시).
6. **종료 시점 안정화** — Logger/ModelManager를 **immortal singleton**(heap allocation + 미해제)으로 변경. ORT 1.26은 process exit 시점에 내부 mutex를 다시 잡는 코드 경로가 있어 정적 destructor 순서와 충돌하면 `std::system_error: mutex lock failed: Invalid argument`로 SIGABRT(134) 발생. 싱글톤이 process 수명 끝까지 살아있으면 모든 ORT 객체도 leak되어 OS가 정리 — exit 코드 0 회복.

**검증**:
- ✅ `git grep -nE 'std::(cout|cerr)' CppLibrary/src/` → `Logger.cpp` 외 0건
- ✅ `grep -nE '^\s*static ' CppLibrary/src/aiRunner.cpp` → 0건
- ✅ `cmake --build build --config Release -j` 통과
- ✅ `./build/CppApplication/CppApplication /tmp/mnist.onnx` 7개 시나리오: 단일/배치/2x/async 단일/async 배치/멀티 인스턴스/로깅 — Step 1과 동일 출력(`-57.3601`, instance2=-1.78341, instance3=-3.5309) + EXIT_CODE=0
- ✅ `--gpu`(CoreML)도 동일 출력 + EXIT_CODE=0
- ✅ 배치 실패 메시지가 `[ERROR] Default batch run error: ...` 형태로 Logger 포맷에 포함(MNIST batch=1 고정은 사전 존재 모델 한정 제약)
- ⏭️ ThreadSanitizer 빌드: 시간상 미실시. Step 3에서 async 경로 재설계 시점에 함께 검증

**수정 파일**:
- `CppLibrary/include/Logger.h`, `CppLibrary/src/Logger.cpp` (C 콜백 브리지 + immortal singleton)
- `CppLibrary/include/ModelManager.h`, `CppLibrary/src/ModelManager.cpp` (default-instance + async 흡수 + immortal singleton)
- `CppLibrary/include/ExecutionProviderConfig.h`, `CppLibrary/src/ExecutionProviderConfig.cpp` (`device_id` 파라미터)
- `CppLibrary/src/Model.cpp` (cout/cerr 제거)
- `CppLibrary/src/Workflow.cpp` (cout/cerr 제거)
- `CppLibrary/src/DataLoader.cpp` (cout/cerr 제거)
- `CppLibrary/src/aiRunner.cpp` (362→230행, static 0개)

**Step 2 후 깨달은 점** (Step 3 작업에 영향):
- 멀티 모델 cache 사본은 무의미 — Step 3의 새 핸들 API도 직접 vector 참조 반환 패턴 유지
- ORT 1.26 exit-time mutex 이슈는 모든 singleton에 적용 — Step 3에서 새 singleton 추가 시 동일 immortal 패턴 적용 필요
- `default_async_future_`를 ModelManager가 보유하는 구조는 Step 3에서 핸들별 async 상태로 자연스럽게 일반화 가능 (Workflow.current_future_를 활용)

### Step 3 — 공개 C ABI 재설계 (Breaking, v0.2.0)

**의도**: 핸들 기반 단일 형태로 통일. v0.2.0 메이저 변경.

작업:
1. `CppLibrary/include/aiRunner.h`에서 전역 상태 API **전부 삭제**:
   - `InitModel`, `RunModelInt/Float/Double`, `GetFlattenedOutput`, `GetOriginalShape`, `GetElementsPerSample`
   - `RunModelBatchInt/Float/Double`
   - `RunModelFloatAsync/IntAsync/DoubleAsync`, `RunModelBatchFloatAsync`, `IsInferenceRunning`, `WaitForInference`, `GetAsyncResult`, `RunModelFloatAsyncCallback`
   - `allCheck`
2. **남는/추가되는 핸들 기반 API** (모두 첫 인자 `int instance_id`, 모두 `bool` 반환):
   - `int CreateModelInstance()` / `bool DestroyModelInstance(int)`
   - `bool InitModelInstance(int, const char* modelPath, bool cpu_use)` — `cpu_use` 의미는 동일하게 유지하거나 `ExecutionProviderConfig`로 옵션화(아래 5번).
   - `bool RunModelInstanceInt/Float/Double(int, T* data, int num_elements)`
   - `bool RunModelInstanceBatchInt/Float/Double(int, T* data, int batch_size, int elements_per_sample)`
   - `bool RunModelInstanceFloatAsync(int, float*, int)` 외 type/batch 변종 + `bool WaitForInstanceInference(int)` + `bool IsInstanceInferenceRunning(int)` + `bool RunModelInstanceFloatAsyncCallback(int, float*, int, InferenceCallbackC, void*)`
   - `const float* GetInstanceFlattenedOutput(int, int* size)`, `const int64_t* GetInstanceOriginalShape(int, int* size)`, `int GetInstanceElementsPerSample(int)`, `bool IsInstanceInitialized(int)`, `int GetModelInstanceCount()`
3. **로깅 API**: 글로벌이 자연스러우므로 핸들화하지 않고 현행 유지(`SetLogLevel`, `EnableLogConsole`, `EnableLogTimestamp`, `SetLogFile`, `CloseLogFile`, `SetLogCallback`, `ClearLogCallback`, `LogDebug/Info/Warn/Error`).
4. **네이밍 정렬**: C ABI는 PascalCase 유지(이미 다수). 내부 C++ 멤버 메서드 `init_model`/`setSessionOption`/`getFlattenedOutput` 등은 일관된 `snake_case`로 통일(예: `init_model`, `set_session_options`, `get_flattened_output`). `Workflow`, `Model`, `ModelManager` 클래스 전반 정리.
5. **EP 선택 노출**(선택, 시간 허용 시): `bool InitModelInstanceWithEP(int, const char*, EasyOnnxEP ep, int device_id)` 같은 확장 — `EasyOnnxEP` enum: `CPU=0, CoreML=1, DML=2`. 시간 부족하면 기존 `cpu_use` 유지하고 후속 작업으로.
6. **소비자 마이그레이션**:
   - `CppApplication/main.cpp` (275줄) 전체를 새 API로 재작성. 시나리오 수와 의미는 그대로 유지(싱글 → 배치 → 성능 비교 → async → async 배치 → 다중 인스턴스 → 로깅).
   - 빌드 후 동일 모델에 대해 출력 동일성 확인.

**완료 기준**:
- `aiRunner.h` 전 함수 시그니처가 핸들 기반 + `bool` 반환 + PascalCase로 통일.
- `CppApplication`이 새 API만 사용해서 모든 시나리오 통과.
- `git grep -n 'workflow' CppLibrary/src/aiRunner.cpp` 결과 0건(전역 변수 잔재 없음).

수정 파일:
- `CppLibrary/include/aiRunner.h` (대대적 재작성)
- `CppLibrary/src/aiRunner.cpp` (대대적 슬림화)
- `CppLibrary/include/{Workflow,Model,ModelManager}.h`, `CppLibrary/src/{Workflow,Model,ModelManager}.cpp` (메서드 네이밍 정렬)
- `CppApplication/main.cpp` (새 API로 재작성)

### Step 4 — 문서·패키징·버전 bump

작업:
1. **버전 bump**: `CMakeLists.txt:2` `VERSION 0.1.0` → `0.2.0`. `README.md`에 호환성 경고 섹션 추가.
2. **README 갱신**: ORT 버전 1.26.0(macOS) + Windows 가속 EP 상태(미정/CPU only) 명시, 새 API 예시로 교체, 마이그레이션 표(`InitModel` → `CreateModelInstance + InitModelInstance` 등) 추가.
3. **`docs/cpp-integration.md`**: 코드 예제를 새 API로 교체(현 라인 64–82). 핸들 라이프사이클 설명 추가.
4. **`docs/unity-integration.md`**: P/Invoke 시그니처 전부 신 API로 교체(현 라인 46–60).
5. **release zip 검증**: `scripts/build_release.sh` / `scripts/build_release.ps1` 실행 → `easyonnx-mac-arm64-v0.2.0.zip`, `easyonnx-windows-x64-v0.2.0.zip` 생성 확인. 내부 `include/aiRunner.h`가 새 시그니처인지 확인.
6. **마이그레이션 가이드**: `docs/migration-v0.2.0.md` 신규 — 표 형식으로 구버전 호출 → 신버전 호출 1:1 매핑. (예: `InitModel("m.onnx", false)` → `int h = CreateModelInstance(); InitModelInstance(h, "m.onnx", false);`)

**완료 기준**: README/docs 모두 새 API와 ORT 1.26.0(macOS) 반영, release zip 두 개 생성(Windows zip은 DML 결정 후), 마이그레이션 가이드 PR.

수정/추가 파일:
- `CMakeLists.txt`, `README.md`
- `docs/cpp-integration.md`, `docs/unity-integration.md`
- `docs/migration-v0.2.0.md` (신규)

---

## Critical Files (reference index)

| 파일 | 역할 | 본 계획에서의 변경 |
|---|---|---|
| `CppLibrary/include/aiRunner.h` | 공개 C ABI | Step 3에서 대대적 재작성 |
| `CppLibrary/src/aiRunner.cpp` (362줄) | C ABI 진입점 + 전역 상태 | Step 2에서 슬림화, Step 3에서 핸들 어댑터만 남음 |
| `CppLibrary/src/Workflow.cpp` (232줄) | 추론 오케스트레이션 | Step 2 cout 제거, Step 3 네이밍 정렬 |
| `CppLibrary/src/Model.cpp` (140줄) | ORT 세션 래퍼 | Step 1 ORT 호환 점검, Step 2 cout 제거, Step 3 네이밍 |
| `CppLibrary/src/ModelManager.cpp` (155줄) | 인스턴스 풀 | Step 2 출력 캐시 흡수, mutex 추가 |
| `CppLibrary/src/Logger.cpp` (151줄) | 로깅 | Step 2에서 호출 빈도 급증, lazy-init 검토 |
| `CppLibrary/src/ExecutionProviderConfig.cpp` (57줄) | EP 와이어링 | Step 1 ORT 호환, Step 3에서 device_id 옵션 노출 (선택) |
| `CppLibrary/CMakeLists.txt` (149줄) | 라이브러리 빌드 | Step 1 ORT 버전 상수 |
| `CppApplication/main.cpp` (275줄) | 종합 테스트 | Step 3에서 새 API로 재작성 |
| `Library/ONNXRuntime/{mac,windows}/` | vendored ORT | Step 1 바이너리 교체 |
| `README.md`, `docs/*.md` | 사용자 문서 | Step 4 갱신 |
| `scripts/build_release.{sh,ps1}` | Release zip | Step 4 검증 |

## Reuse / Don't Reinvent

- **`ModelManager` 싱글톤**(`ModelManager.cpp:60`의 `std::map<int, std::shared_ptr<Workflow>> instances_`)은 이미 잘 작성됨. Step 2의 전역 상태 흡수는 이 자료구조 확장으로 처리.
- **`Logger`**(`include/Logger.h`)의 콜백/파일/레벨 인터페이스는 그대로. 호출 측만 일관화.
- **`ExecutionProviderConfig::TryAppendAcceleratedEP`**의 플랫폼 분기 구조는 그대로 두고 시그니처만 확장.
- **release 패키징**(`scripts/build_release.{sh,ps1}`)은 검증만 하고 재구성하지 않음.
- 이미 작성된 `docs/refactor/2026-05-20-macos-arm64-port.md`의 phased 사고방식을 본 계획에도 그대로 차용.

## Verification (end-to-end)

각 Step 끝마다 수동 회귀:

```bash
# Step 1, 2, 3 공통
cmake -B build -S .
cmake --build build --config Release -j
./build/CppApplication/CppApplication path/to/test_model.onnx        # CPU 경로
./build/CppApplication/CppApplication path/to/test_model.onnx --gpu  # CoreML(mac) / DML(win)

# Step 2 추가: ThreadSanitizer
cmake -B build-tsan -S . -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1"
cmake --build build-tsan -j
./build-tsan/CppApplication/CppApplication path/to/test_model.onnx   # async/멀티 시나리오에서 race 보고 0건

# Step 4 추가: release zip
./scripts/build_release.sh                                            # macOS
unzip -l easyonnx-mac-arm64-v0.2.0.zip                                # include/aiRunner.h가 새 시그니처인지 확인
```

CppApplication의 7개 시나리오(`main.cpp:38-270`)는 매 Step 후 통과해야 함. 출력 수치는 Step 1 직후 한 번 기록해두고 이후 Step에서 비교.

## Out of Scope (명시)

- Linux 빌드/번들링
- 신규 EP(CUDA, TensorRT, OpenVINO)
- GitHub Actions CI 자동화
- 코드 서명/Notarization
- Intel macOS / Universal 바이너리
- ORT 1.18.1로의 다운그레이드 호환 어댑터
- v0.1.x 시리즈 보안 패치 백포트
