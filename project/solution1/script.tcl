############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project project
set_top main
add_files hls_src/accelerator.cpp
add_files hls_src/accelerator.h
add_files hls_src/tb.cpp
add_files -tb hls_src/tb.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
#source "./project/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
