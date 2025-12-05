#!/bin/bash
rm -rf ./cpp/build/*
rm -rf ./cpp/chessrl_inference_cpp.egg-info/*
rm -f ./cpp/*.so  # Clean up any stale .so files that could shadow the installed package
uv pip install --force-reinstall ./cpp