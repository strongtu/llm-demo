set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install llama-cpp-python[server] --force-reinstall --upgrade --no-cache-dir --verbose