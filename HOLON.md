---
# Holon Identity v1
uuid: "c7a1e0f3-5b2d-4e8a-9f6c-3d1a7b4e2c8f"
given_name: "Wisupaa"
family_name: "Whisper"
motto: "Every sound becomes data."
composer: "B. ALTER"
clade: "probabilistic/perceptual"
status: draft
born: "2026-03-02"

# Lineage
parents: []
reproduction: "assisted"

# Metadata
generated_by: "codex"
proto_status: defined
---

# Wisupaa Whisper

> *"Every sound becomes data."*

## Description

General-purpose whisper.cpp holon. Exposes the full whisper.cpp feature
set (transcription, language detection, token-level timestamps, VAD,
model introspection) as JSON-RPC services over the `cpp-holons` SDK
transport layer.

This is a **probabilistic/perceptual** holon — downstream holons compose
its RPCs to build specialized pipelines (alignment, live transcription, etc.).

## Contract

- Proto: `protos/whisper/v1/whisper.proto`
- Service: `whisper.v1.Whisper`
- Transport: `stdio://` (default), `tcp://`, `unix://`

## Introspection Notes

- whisper.cpp lives in `third_party/whisper.cpp/` (git submodule).
- The `cpp-holons` SDK is in the parent monorepo: `../../organic-programming/sdk/cpp-holons/`.
- Model files (`.bin`) are loaded at runtime via model path argument.
- Audio input: 16 kHz, mono, float32 PCM.
