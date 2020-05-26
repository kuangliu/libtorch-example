rm -rf build

cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/liukuang/apps/libtorch/libtorch

cmake --build build
