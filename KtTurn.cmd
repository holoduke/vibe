@echo off
cd /d C:\Users\gillis\dev\quake-like
set VIBE_NO_NET=1
set VIBE_TEST_SIZE=1280x720
del kt_done.txt 2>nul
set VIBE_CFG=ka_lotlow.cfg
set VIBE_TEST_JEEP=1.0,0
set VIBE_TEST_JEEP_AT=1689.5,837.2,3.14159
set VIBE_TEST_CAR=1
set VIBE_TEST_JEEP_TURN=4
build\vibe.exe --frames 560 --log kt_s4.log
set VIBE_TEST_JEEP_TURN=7
build\vibe.exe --frames 760 --log kt_s7.log
set VIBE_TEST_CAR=0
set VIBE_TEST_JEEP_TURN=6
build\vibe.exe --frames 700 --log kt_j6.log
echo done > kt_done.txt
