rm -r build
rm -r bin
cmake \
    -S .\
    -B ./build \
    ..

cd build
make -j12
#make install
