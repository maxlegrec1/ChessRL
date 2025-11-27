#!/bin/bash
rm ./cpp/build/*
rm ./cpp/chessrl_inference_cpp.egg-info/*
uv pip install --force-reinstall ./cpp