@echo off
pushd %~dp0
swig -python -doxygen bindings.i
if errorlevel 1 (
    echo SWIG failed
    popd
    exit /b 1
)
python compile_bindings.py build_ext --inplace
if errorlevel 1 (
    echo Build failed
    popd
    exit /b 1
)
rd /s /q build
del /f /q bindings_wrap.c
popd