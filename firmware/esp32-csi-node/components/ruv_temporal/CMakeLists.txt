# ESP-IDF component manifest for the ruv_temporal Rust staticlib (ADR-095).
#
# Build flow:
#   1. Run `cargo +esp build --release --target xtensa-esp32s3-none-elf` in
#      this directory. Output: target/xtensa-esp32s3-none-elf/release/libruv_temporal.a
#   2. Register the resulting static library and the public header dir
#      with idf_component_register so it shows up on the firmware's
#      include path and link line.
#
# Phase 4: scaffold only — registered but no kernel work runs yet.
# Phase 5: cross-compile validated, binary delta measured.
# Phase 6: enabled via CONFIG_CSI_TEMPORAL_HEAD_ENABLED Kconfig flag and
#          fed from edge_processing.c.

set(RUV_TEMPORAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(RUV_TEMPORAL_TARGET "xtensa-esp32s3-none-elf")
set(RUV_TEMPORAL_PROFILE "release")
set(RUV_TEMPORAL_LIB
    "${RUV_TEMPORAL_DIR}/target/${RUV_TEMPORAL_TARGET}/${RUV_TEMPORAL_PROFILE}/libruv_temporal.a")

# Run the cargo build as a custom command. ESP-IDF's CMake runs at
# configure time; we want the staticlib to exist before idf_component_register
# runs so it can be added as INTERFACE_LINK_LIBRARIES.
add_custom_command(
    OUTPUT "${RUV_TEMPORAL_LIB}"
    WORKING_DIRECTORY "${RUV_TEMPORAL_DIR}"
    COMMAND cargo +esp build --release --target ${RUV_TEMPORAL_TARGET}
    COMMENT "Building ruv_temporal Rust staticlib for ${RUV_TEMPORAL_TARGET}"
    VERBATIM
)
add_custom_target(ruv_temporal_rust_build ALL DEPENDS "${RUV_TEMPORAL_LIB}")

idf_component_register(
    SRCS "shim.c"                # tiny C shim so idf_component_register has a SRCS
    INCLUDE_DIRS "include"
    PRIV_REQUIRES "esp_common"
)

# Wire the staticlib in.
add_dependencies(${COMPONENT_LIB} ruv_temporal_rust_build)
target_link_libraries(${COMPONENT_LIB} INTERFACE "${RUV_TEMPORAL_LIB}")
