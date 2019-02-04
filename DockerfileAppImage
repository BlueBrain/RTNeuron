FROM bluebrain/rtneuron_builder

# RTNeuron
################

ADD . /RTNeuron
# Avoid linking against libX11.so as it depends on dev packages to exist.
RUN cd /RTNeuron && mkdir build && cd build && cmake ../ -DCLONE_SUBPROJECTS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && make -j mvd-tool RTNeuron-all RTNeuron-install

#################
RUN mkdir -p /root/RT
RUN mkdir -p ./usr/
WORKDIR /root/RT
RUN /usr/local/bin/python3.7 -m venv --copies usr
RUN rsync -av --exclude site-packages /usr/local/lib/python3.7/ /root/RT/usr/lib64/python3.7/

RUN source  ./usr/bin/activate && pip install pyopengl numpy PyQt5==5.9.2 ipython
ENV PYTHON_SITE_PACKAGES=usr/lib/python3.7/site-packages/
RUN cp -r /usr/local/lib/python3.7/site-packages/rtneuron/ ${PYTHON_SITE_PACKAGES} && \
  cp -r /usr/local/lib/python3.7/site-packages/brain/ ${PYTHON_SITE_PACKAGES}

RUN cp -Pr /usr/local/lib64/lib*so* /usr/local/lib/lib*so* /usr/lib64/libGLEWmx* /usr/lib64/libturbojpeg.so* /usr/lib64/libpng*.so* /usr/lib64/libjpeg*.so* usr/lib
RUN mkdir usr/lib/osgPlugins-3.4.1 && for ext in `echo 3ds bmp ive jpeg lwo obj osg ply png pnm stl tga tiff`; do cp /usr/local/lib64/osgPlugins-3.4.1/osgdb_$ext.so usr/lib/osgPlugins-3.4.1; done
RUN cp /usr/lib64/libjpeg.so* /usr/lib64/libpng*.so* /usr/lib64/libtiff.so* /usr/lib64/libjbig* usr/lib

RUN mkdir -p usr/share/RTNeuron && cp -rf /usr/local/share/RTNeuron/shaders usr/share/RTNeuron
RUN mkdir -p usr/share/osgTransparency && cp -rf /usr/local/share/osgTransparency/shaders usr/share/osgTransparency
RUN cp -r /usr/local/bin/rtneuron usr/bin

# APP image
###################
WORKDIR /root
RUN curl -LO https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage && \
chmod +x ./linuxdeploy-x86_64.AppImage

# AppImageTool
RUN curl -LO https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
RUN chmod +x /root/appimagetool-x86_64.AppImage
RUN /root/appimagetool-x86_64.AppImage --appimage-extract

WORKDIR /root/RT

ADD packaging/AppImage/config .
RUN chmod +x /root/RT/AppRun

CMD ["/root/squashfs-root/AppRun", "/root/RT", "/tmp/output/rtneuron_x86_64.AppImage"]
