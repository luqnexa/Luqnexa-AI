#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
set -eu
cd "$(dirname "$0")/.."
cargo build -p nextime-ffi --release --locked
ffi_target=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
cc -std=c11 -Wall -Wextra -Werror -Icrates/nextime-ffi/include \
  platforms/common/ffi-demo.c -L"$ffi_target/release" -lnextime_ffi \
  -Wl,-rpath,"$ffi_target/release" -o "$ffi_target/release/nextime-ffi-demo"
"$ffi_target/release/nextime-ffi-demo"
