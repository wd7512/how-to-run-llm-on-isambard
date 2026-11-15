#!/bin/bash
# Shared settings, sourced by every script. Edit MODEL / IMAGE here only.
# $SCRATCHDIR is set for you on Isambard-AI (see brics/userenv); jobs inherit it.
: "${SCRATCHDIR:?SCRATCHDIR is not set - run on an Isambard-AI login node or load brics/userenv}"

export MODEL_REPO="deepseek-ai/DeepSeek-V4.1-Flash-1017"
export MODEL_DIR="$SCRATCHDIR/models/DeepSeek-V4.1-Flash-1017"

# Pinned arm64 image (vllm/vllm-openai:latest as pulled on 2026-11-15: vLLM 0.33.0,
# torch 2.14, CUDA 13.0). Tags move, the digest does not.
export IMAGE_REF="docker://vllm/vllm-openai@sha256:e3481489818d3e563cae517b84ab959cc8b69f85a9fbb771c189140131ffc65c"
export SIF="$SCRATCHDIR/containers/vllm-0.33.0-arm64.sif"

export HF_VENV="$SCRATCHDIR/envs/hfdl"
export HF_HOME="$SCRATCHDIR/hf"
