#!/bin/bash

cd cpp
uv run python setup.py build_ext --inplace
cd ..