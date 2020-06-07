# [rust-opencv](https://github.com/fbielejec/rust-opencv)

A primer on using [OpenCV](https://opencv.org/opencv-4-0/) with [Rust](https://www.rust-lang.org/).

## Documentation

- [Rust bindings for OpenCV - repository and examples](https://github.com/twistedfall/opencv-rust/)
- [Rust bindings for OpenCV - documentation](https://docs.rs/opencv/0.33.0/opencv/)
- [OpenCV C++ documentation](https://docs.opencv.org/master/)

## Development

### Install development tools
```bash
sudo apt-get install build-essential cmake unzip pkg-config
```

### Install image and video IO libraries
```
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libxvidcore-dev libx264-dev libgtk2.0-dev
```

### Install math libraries
```
sudo apt-get install libatlas-base-dev gfortran
```

### Download OpenCV 4
Get the latest version:
```bash
wget https://github.com/opencv/opencv/archive/4.2.0.zip ~/Downloads
```

Create a directory and unzip it:
```bash
sudo mkdir /opt/opencv
sudo unzip ~/Downloads/opencv-4.2.0.zip -d /opt/opencv
```

Create symlink `latest` pointing to the current version of OpenCV:
```bash
cd /opt/opencv
sudo ln -s opencv-4.2.0 latest
```

In the future you can unlink it and create new symlink with:
```bash
sudo unlink /opt/opencv/latest
sudo ln -s opencv-X.X.X latest
```

### Compile OpenCV
Create a temporary build directory:
```bash
cd /opt/opencv/latest
sudo mkdir release
```

Generate build files:
```
cd /opt/opencv/latest/release
sudo cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=YES -DCMAKE_INSTALL_PREFIX=/opt/opencv/opencv-4.2.0  /opt/opencv/opencv-4.2.0
```

Build and install:
```bash
cd /opt/opencv/latest/release
sudo make -j4
sudo make install
```

Update links/cache used by the dynamic loader:
```bash
sudo ldconfig
```

### Setup PATH env variables
Add to .bashrc:
```
export PATH=/opt/opencv/latest/bin:/opt/opencv/latest/release/bin:${PATH}
export LD_LIBRARY_PATH=/opt/opencv/latest/release/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/opt/opencv/latest/lib/pkgconfig
export OPENCV_TEST_DATA_PATH=/opt/opencv/latest/opencv_extra-master/testdata
```

Or if your're using fish:
```fish
# OpenCV paths
set -gx PATH /opt/opencv/latest/bin $PATH
set -gx PATH /opt/opencv/latest/release/bin $PATH
set -gx LD_LIBRARY_PATH /opt/opencv/latest/release/lib $LD_LIBRARY_PATH
set --export PKG_CONFIG_PATH /opt/opencv/latest/lib/pkgconfig
set --export OPENCV_TEST_DATA_PATH /opt/opencv/latest/opencv_extra-master/testdata
```
### Run test suite

You need to export the `OPENCV_TEST_DATA_PATH` env variable and copy two files named `lena.jpg` and `lena.png` to the release directory
```bash
export OPENCV_TEST_DATA_PATH=/opt/opencv/latest/opencv_extra-master/testdata
sudo cp ~/Dropbox/Rust/opencv/lena.png /opt/opencv/latest/release/bin/lena.png
sudo cp ~/Dropbox/Rust/opencv/lena.jpg /opt/opencv/latest/release/bin/lena.jpg
cd /opt/opencv/latest/release
sudo ./bin/opencv_test_core
```
