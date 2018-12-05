FROM centos:7.0.1406

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

# Needed to install libopenssl
RUN yum -y swap -- remove fakesystemd -- install systemd systemd-libs 
RUN yum install -y yum-plugin-ovl
RUN yum install -y zlib-devel bzip2 bzip2-devel flex bison git tar gcc gcc-c++ make mesa-libGL-devel libpng-devel libjpeg-turbo-devel libtiff-devel readline-devel sqlite-devel openssl-devel gdbm-devel libtool libffi-devel qt5-base-devel qt5-qtquickcontrols2-devel glew-devel turbojpeg-devel turbojpeg yum-plugin-ovl  gmp-devel mpfr-devel libmpc-devel 
RUN yum update -y nss nss-util nspr curl
RUN yum clean all 

# GCC 5.4.0
################

RUN curl ftp.gnu.org/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.bz2 -O && \
tar xvfj gcc-5.4.0.tar.bz2 && \
mkdir gcc-5.4.0/build  && cd  gcc-5.4.0/build && \
../configure --enable-languages=c,c++ --disable-multilib && \
make -j && make install && rm -rf /gcc-5.4.0

ENV CXX=/usr/local/bin/g++
ENV CC=/usr/local/bin/gcc 

# CMake
################

# Need to build this from source, since the binaries from Kitware don't
# run with the old version of the libraries in this Centos version
RUN curl -OL https://cmake.org/files/v3.12/cmake-3.12.2.tar.gz && tar xfz cmake-3.12.2.tar.gz && cd cmake-3.12.2 && ./configure && make -j6 install
RUN rm -rf cmake-3.12.2*

# HDF5
################

RUN curl -OL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.3/src/hdf5-1.10.3.tar.bz2 && tar xfj hdf5-1.10.3.tar.bz2 && cd hdf5-1.10.3 && ./configure --prefix=/usr/local --enable-build-mode=production && make -j6 install
RUN rm -rf hdf5-1.10.3*

# Doxygen
################

# Needed for docstrings generation and system doxygen is too old
# after 1.8.6, we can't use anymore system flex, see https://stackoverflow.com/questions/20844635
RUN curl -OL https://github.com/doxygen/doxygen/archive/Release_1_8_5.tar.gz && tar xfz Release_1_8_5.tar.gz && cd doxygen-Release_1_8_5 && ./configure && make -j6 && make install
RUN rm -rf Release_1_8_5.tar.gz doxygen-Release_1_8_5

# Python 3.7
################

RUN curl -OL https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && tar xfz Python-3.7.0.tgz && cd Python-3.7.0 && ./configure --enable-optimizations --enable-shared && make -j 6 && make install
RUN find /usr/local/lib/python3.7/ | grep "\(__pycache__\|/test$\)$" | xargs rm -rf
RUN rm -rf /usr/local/lib/python3.7/tests Python-3.7.0*
RUN pip3 install pyopengl numpy sphinx==1.3.6 PyQt5==5.9.2 ipython lxml
# Needed for building Boost.Python
RUN ln -s /usr/local/include/python3.7m /usr/local/include/python3.7

# Boost
################

RUN curl -OL https://downloads.sourceforge.net/project/boost/boost/1.65.0/boost_1_65_0.tar.gz && tar xfz boost_1_65_0.tar.gz
# Patching boost source code
RUN cd boost_1_65_0 && sed -i -e "s/_PyUnicode_AsString/(void*)_PyUnicode_AsString/" libs/python/src/converter/builtin_converters.cpp
RUN cd boost_1_65_0 && ./bootstrap.sh --with-libraries=date_time,iostreams,filesystem,program_options,regex,serialization,system,test,thread,python --with-python=python3 && ./b2 -s -j6 -q threading=multi --build-type=minimal install
RUN rm -rf boost_1_65_0*

 #$OpenSceneGraph
###############
RUN env |grep -i CXX
RUN curl -OL https://github.com/openscenegraph/OpenSceneGraph/archive/OpenSceneGraph-3.4.1.tar.gz && tar xfz OpenSceneGraph-3.4.1.tar.gz && cd OpenSceneGraph-OpenSceneGraph-3.4.1 && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j6 && make -j6 install
RUN rm -rf OpenSceneGraph*


