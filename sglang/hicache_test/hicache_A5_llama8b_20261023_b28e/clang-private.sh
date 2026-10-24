#!/bin/bash
# Prefer the container's own libc libraries over the host /usr/lib64 mount.
exec /usr/bin/clang++ -L/lib/aarch64-linux-gnu "$@"
