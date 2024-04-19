REM ..\llama.cpp\build\bin\Release\main.exe -m "..\llama.cpp\models\vicuna-13b-v1.5-16k.Q4_K_M.gguf" --n-predict 512 --repeat_penalty 1.0 --color -i -r "User:" -f "..\llama.cpp\prompts\chat-with-bob.txt"
    ..\llama.cpp\build\bin\Release\main.exe -m "..\llama.cpp\models\chinese-alpaca-2-13b-16k.Q4_K.gguf" --n-predict 512 --repeat_penalty 1.0 --color -i -r "User:" -f "..\llama.cpp\prompts\chat-with-bob.txt" --n-gpu-layers 128
REM ./main --color --interactive --model <MODELS_DIRECTORY>/7B/ggml-model-q4_0.bin --n-predict 512 --repeat_penalty 1.0 --n-gpu-layers 15000 --reverse-prompt "User:" --in-prefix " " -f prompts/chat-with-bob.txt
REM --n-predict 512         -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
REM --n-gpu-layers 128      -ngl N, --n-gpu-layers N      number of layers to store in VRAM
REM -n 256                  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
REM --repeat_penalty 1.0    --repeat-penalty N    penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)
