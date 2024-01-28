# Use Ubuntu as the base image
FROM ubuntu:latest as build

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    meson \
    ninja-build \
    zlib1g-dev

# Clone lc0 repository
WORKDIR /lc0
# RUN git clone --recurse-submodules https://github.com/LeelaChessZero/lc0.git /lc0
COPY . .

# Compile lc0
RUN meson build . --buildtype release
RUN ninja -C build

# Cleanup to reduce image size
RUN apt-get remove --purge -y git meson ninja-build && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


FROM scratch
COPY --from=build /lc0/build/lc0 .
ENTRYPOINT ["/lc0"]
