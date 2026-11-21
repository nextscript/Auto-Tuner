# Legacy Voice Lab (tc-mb/llama.cpp-omni) Vulkan build for VoxCPM2.
# Note: mainline llama.cpp b12531+ (PR #32443) also includes native VoxCPM2
# support in llama-tts / llama-server (/tts) via voxcpm2_gen / voxcpm2_spkenc.
$ErrorActionPreference = 'Stop'
$src = 'L:/LAB/ai-local/voicelab_llama.cpp-omni'
if (!(Test-Path "$src/CMakeLists.txt")) { throw 'Clone tc-mb/llama.cpp-omni into the dedicated Voice Lab directory first.' }
$build = Join-Path $src 'build-voicelab'
& cmake -S $src -B $build -G 'Visual Studio 18 2026' -A x64 -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CCACHE=OFF -DLLAMA_OPENSSL=OFF
if ($LASTEXITCODE) { throw 'VoxCPM configure failed' }
& cmake --build $build --config Release --parallel 20 --target voxcpm2-cli llama-tts-server
if ($LASTEXITCODE) { throw 'VoxCPM build failed' }
