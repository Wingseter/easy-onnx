# Plan: easy-onnx — Apple Silicon macOS 지원 + EP 와이어링 정비

## Context

`easy-onnx`는 ONNXRuntime을 감싼 C++ 추론 라이브러리로, 원래 Windows DLL 위주로 개발됐다. README는 macOS도 지원한다고 명시하지만 사용자는 아직 mac에서 빌드 시도조차 해본 적이 없는 상태다. CMake 분기·헤더·dylib(arm64)는 이미 동봉되어 있어 인프라 기반은 있다.

동시에 코드를 들여다보니 README의 또 다른 주장 — Windows에서 DirectML 가속을 쓴다 — 도 사실은 미구현이었다. `Model::setSessionOption(bool cpu_use)`는 스레드 옵션만 만지고 EP를 전혀 붙이지 않는다. 즉 `cpu_use=false`로 호출해도 양쪽 OS 모두 그냥 CPU로 도는 상태.

이번 작업의 목표:
- Apple Silicon mac(arm64)에서 처음부터 빌드 → CppApplication 실행 → CoreML EP까지 실제로 동작.
- 같은 김에 Windows 쪽 DirectML EP 와이어링도 같이 마무리(코드 일관성).
- `UnityApplication/` 폴더는 제거하고 Unity 연동은 `docs/unity-integration.md` 한 문서로만 남김.
- 각 OS에서 `scripts/build_release.{sh,ps1}` 한 줄 실행 → `easyonnx-<os>-<arch>-vX.Y.Z.zip` 산출.
- CI/코드 서명/노타라이즈는 이번 범위 밖(별도 이슈).

## Critical files

- `CppLibrary/src/Model.cpp` — EP append 호출 추가, 디버그 `"what?"` 출력 제거.
- `CppLibrary/Utils/pch.h` — CoreML/DML 헤더 include를 플랫폼 가드.
- `CppLibrary/src/ExecutionProviderConfig.cpp` (신규) — 플랫폼별 EP append 로직 격리.
- `CppLibrary/include/ExecutionProviderConfig.h` (신규) — 위 함수의 헤더.
- `CppLibrary/CMakeLists.txt` — rpath 설정, onnxruntime dylib 복사, `install()` 룰 추가, 신규 소스 추가.
- `CppApplication/CMakeLists.txt` — rpath 설정, onnxruntime dylib 실행 파일 옆에 복사.
- `CppApplication/main.cpp` — 하드코딩된 절대 경로를 `argv[1]`로 변경, usage 메시지 추가.
- `CMakeLists.txt` (root) — `project(easy-onnx VERSION 0.1.0 LANGUAGES CXX)`, mac일 때 `CMAKE_OSX_ARCHITECTURES=arm64` 강제.
- `scripts/build_release.sh` (신규) — mac arm64 zip 생성.
- `scripts/build_release.ps1` (신규) — Windows zip 생성.
- `docs/cpp-integration.md` (신규) — release zip 소비자용 가이드.
- `docs/unity-integration.md` (신규) — P/Invoke 예제 (현재 README 코드 기반).
- `README.md` — 새 구조 반영, EP 동작 매트릭스 표.
- `UnityApplication/` — 폴더 통째로 삭제.
- `debugging/CMakeLists.txt` — `OnnxRuntime`→`ONNXRuntime` 케이스 오타 수정 (원샷 작업이라 같이).

## Approach

### Phase 1 — 빌드 위생 (mac에서 일단 빌드/링크/실행이 깨끗하게 되게)

Mac dylib의 install_name은 `@rpath/libonnxruntime.1.18.1.dylib`이므로 소비자가 rpath를 가져야 한다.

- `CppLibrary/CMakeLists.txt`의 `aiRunner` 타겟에:
  - `set_target_properties(aiRunner PROPERTIES MACOSX_RPATH ON BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")`
  - post-build로 onnxruntime dylib(`libonnxruntime.1.18.1.dylib` + `libonnxruntime.dylib` 심볼릭) 을 `$<TARGET_FILE_DIR:aiRunner>`와 `Library/aiRunner/mac/lib/` 양쪽에 복사. (Windows 분기: `onnxruntime.dll` 복사)
- `CppApplication/CMakeLists.txt`의 `CppApplication` 타겟에:
  - `set_target_properties(CppApplication PROPERTIES INSTALL_RPATH "@executable_path" BUILD_WITH_INSTALL_RPATH FALSE)` + `BUILD_RPATH "@executable_path"`.
  - post-build로 `libaiRunner.dylib` + onnxruntime dylib를 실행 파일 옆에 복사.
- root `CMakeLists.txt`:
  - `project(easy-onnx VERSION 0.1.0 LANGUAGES CXX)` (이미 있는 `project()` 줄 교체).
  - `if(APPLE) set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE) endif()`.
  - `EASYONNX_VERSION` 변수를 서브디렉토리/스크립트가 읽을 수 있게 expose.

**검증 게이트(Phase 1):**
```
cmake -B build -S .
cmake --build build --config Release
otool -L build/CppLibrary/libaiRunner.dylib    # @rpath/libonnxruntime.1.18.1.dylib 확인
otool -l build/CppLibrary/libaiRunner.dylib | grep -A2 LC_RPATH    # @loader_path 확인
./build/CppApplication/CppApplication <path-to-some.onnx>   # cpu_use=true로 정상 종료
```

### Phase 2 — CppApplication CLI 정리

`main.cpp`의 하드코딩된 `/Users/kwon/Workspace/.../best_metric_0.7747.onnx`를 `argv[1]`로 바꾸고, `argc < 2`면 `Usage: CppApplication <model.onnx>` 출력 후 종료. 나머지 데모 로직(single/batch/async/multi-model/logging)은 그대로 유지. **Phase 1과 같이 진행해야 Phase 1 검증이 가능**.

### Phase 3 — EP 와이어링 (핵심 변경)

EP append 로직을 별도 파일로 격리해서 `Model.cpp`에 플랫폼 ifdef가 새지 않도록 한다.

- **신규 `CppLibrary/include/ExecutionProviderConfig.h`**:
  ```cpp
  // 헤더에는 플랫폼 분기 없음
  bool TryAppendAcceleratedEP(Ort::SessionOptions& opts);
  ```
- **신규 `CppLibrary/src/ExecutionProviderConfig.cpp`**:
  - `#if defined(__APPLE__)`: `OrtSessionOptionsAppendExecutionProvider_CoreML(opts, 0)` 호출 (flags=0 = ANE+GPU+CPU 사용, dynamic shape 허용, NeuralNetwork format — 가장 안전한 기본값).
  - `#elif defined(_WIN32)`: `OrtSessionOptionsAppendExecutionProvider_DML(opts, 0)` (device 0).
  - `#else`: 아무것도 안 함, `false` 반환.
  - 모두 `try` 블록으로 감싸고 `Ort::Exception` 잡아서 `Logger::error(...)` 로깅 후 `false` 리턴(부른 쪽이 CPU fallback). 성공 시 `Logger::info("EP appended: CoreML"/"DML")` + `true`.
- **`CppLibrary/Utils/pch.h`** 수정:
  - `<coreml_provider_factory.h>` include를 `#ifdef __APPLE__` 가드로.
  - `<dml_provider_factory.h>` include는 `#ifdef _WIN32` 가드로 추가. (이 파일은 d3d12 의존성이 있을 수 있어 반드시 가드 필요)
  - `#ifdef PLATFORM_WINDOWS #pragma comment(lib, ...) #endif`는 `_WIN32`로 정정(현 매크로는 정의된 적이 없어 죽은 코드).
- **`CppLibrary/src/Model.cpp`** 수정:
  - `setSessionOption(bool cpu_use)`에서 기존 thread/parallel 설정 뒤에:
    ```cpp
    if (!cpu_use) {
        TryAppendAcceleratedEP(session_options);  // 실패해도 CPU로 진행
    }
    ```
  - line 18의 `std::cout << "what?" << std::endl;` 삭제.
- **`CppLibrary/CMakeLists.txt`**: `add_library(aiRunner SHARED ...)`의 소스 리스트에 `src/ExecutionProviderConfig.cpp` 추가.

**검증 게이트(Phase 3):**
```
./build/CppApplication/CppApplication <model.onnx>     # 기본 cpu_use=true 그대로
# main.cpp를 잠시 수정하거나 InitModel(...,false) 호출 경로로 CoreML 활성화 확인
# 로그에 "EP appended: CoreML" 출력 + 추론 결과 동일
```

### Phase 4 — Repo 정리 + 문서

- `UnityApplication/` 폴더 전체 삭제. (root CMakeLists에 add_subdirectory 참조 없음 — 이미 확인됨)
- `docs/cpp-integration.md` 신규: release zip 압축 해제 후 CMake 소비자 프로젝트에서 `target_include_directories`/`target_link_directories`/`target_link_libraries`로 `aiRunner` 링크하는 예제. Gatekeeper quarantine 우회법(`xattr -dr com.apple.quarantine <dir>`)도 함께.
- `docs/unity-integration.md` 신규: 현 README의 P/Invoke 스니펫 + Unity Editor의 macOS 플러그인 배치(`Assets/Plugins/macOS/libaiRunner.dylib`, meta에서 CPU=AnyCPU 또는 ARM64 설정) + Windows 배치(`Assets/Plugins/x86_64/aiRunner.dll`) 설명.
- `README.md` 정리: 새 구조 안내, "Platform support" 표(`Windows x64: DirectML`, `macOS arm64: CoreML`), `docs/` 링크, hardcoded Unity 코드 블록은 docs로 이전했음을 안내.
- `debugging/CMakeLists.txt`: `OnnxRuntime` → `ONNXRuntime` 케이스 오타 한 줄 수정.

### Phase 5 — Release 패키징

- `CppLibrary/CMakeLists.txt`에 추가:
  ```cmake
  install(TARGETS aiRunner
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION lib)
  install(DIRECTORY include/ DESTINATION include)
  if(APPLE)
      install(FILES
          ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.1.18.1.dylib
          ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib
          DESTINATION lib)
  elseif(WIN32)
      install(FILES ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll DESTINATION lib)
  endif()
  install(FILES ${ROOT_DIR}/README.md ${ROOT_DIR}/LICENSE DESTINATION .)
  ```
- `scripts/build_release.sh` (mac, 신규):
  ```sh
  #!/usr/bin/env bash
  set -euo pipefail
  VERSION=$(grep -E 'project\(easy-onnx VERSION' CMakeLists.txt | sed -E 's/.*VERSION ([0-9.]+).*/\1/')
  STAGE="stage/easyonnx-mac-arm64"
  rm -rf build "$STAGE"
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/$STAGE"
  cmake --build build --config Release -j
  cmake --install build
  ditto -c -k --sequesterRsrc --keepParent "$STAGE" "easyonnx-mac-arm64-v${VERSION}.zip"
  echo "Built: easyonnx-mac-arm64-v${VERSION}.zip"
  ```
- `scripts/build_release.ps1` (Windows, 신규): 동일 로직, `Compress-Archive` 사용. (이번 PR에서 작성하되 실제 동작 확인은 Windows 머신 차후)
- 산출물 zip 레이아웃:
  ```
  easyonnx-mac-arm64/
    lib/
      libaiRunner.dylib
      libonnxruntime.1.18.1.dylib
      libonnxruntime.dylib -> libonnxruntime.1.18.1.dylib
    include/
      aiRunner.h
      Workflow.h
      Model.h
      ... (CppLibrary/include/*)
    README.md
    LICENSE
  ```

## Verification (end-to-end)

Mac에서 (이번 작업 머신):
1. `rm -rf build && cmake -B build -S . && cmake --build build --config Release` → 에러 없이 종료.
2. `otool -L build/CppLibrary/libaiRunner.dylib`가 `@rpath/libonnxruntime.1.18.1.dylib`를 가리키고, `otool -l ... | grep -A2 LC_RPATH`로 `@loader_path` 확인.
3. 작은 ONNX 모델로 `./build/CppApplication/CppApplication <model.onnx>` 실행 → 정상 종료.
4. `main.cpp`에서 `InitModel(..., false)`로 호출하는 경로 추가 또는 별도 토글로 CoreML 경로 검증 → 로그에 `EP appended: CoreML` + 동일 출력 확인. (CoreML이 모델 일부 ops 미지원해도 fallback으로 crash 없이 진행되어야 함.)
5. `./scripts/build_release.sh` 실행 → `easyonnx-mac-arm64-v0.1.0.zip` 생성. unzip 후 `lib/`/`include/` 구조 확인. 압축 푼 폴더에서 임시 소비자 프로젝트로 링크 시도(`g++ test.cpp -I.../include -L.../lib -laiRunner -Wl,-rpath,@loader_path/../lib`) → 빌드 + 실행 성공.

Windows에서는 이번 머신에서 검증 불가 — 코드는 작성하되 "Windows 검증은 별도 이슈"로 PR에 명시.

## Flagged risks / out of scope

- **DirectML.dll 번들링**: 현재 `Library/ONNXRuntime/windows/lib/`에 `DirectML.dll`이 없다. Windows 10 1903+ 시스템에 보통 있지만 onnxruntime 1.18.1이 요구하는 버전과 다를 수 있음. `docs/cpp-integration.md`에 "Windows 측은 시스템 DirectML.dll 의존, 미작동 시 redistributable 별도 설치" TODO로 명시. 이번 PR에서 dll 동봉은 안 함.
- **Gatekeeper / 코드 서명**: 미서명 dylib을 zip으로 배포하면 수신자 macOS에서 quarantine 경고. `docs/cpp-integration.md`에 `xattr -dr com.apple.quarantine <dir>` 우회법 문서화. 정식 서명/notarize는 별도 이슈.
- **CI**: 이번 범위 밖. 수동 스크립트만 정비.
- **CoreML flag 튜닝**: `flags=0` 기본값 사용. ANE-only(`COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE`)·MLProgram 등은 추후 옵션화.
- **Logger 통합**: `Model.cpp` / `Workflow.cpp`의 다른 `std::cout` 디버그 출력들은 이번에는 그대로 둠(별도 정리 이슈). 단 line 18 `"what?"` 한 줄은 명백한 잔여물이라 제거.
- **debugging/ 서브프로젝트**: root CMakeLists에서 주석 처리된 상태(`# add_subdirectory(debugging)`). 케이스 오타만 고치고 활성화는 안 함.
