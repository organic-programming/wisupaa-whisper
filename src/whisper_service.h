#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace whisper_service {

struct Token {
    std::int32_t id = 0;
    std::string text;
    float p = 0.0f;
    float plog = 0.0f;
    std::int64_t t0 = 0;
    std::int64_t t1 = 0;
    std::int64_t t_dtw = 0;
    float vlen = 0.0f;
};

struct Segment {
    std::int32_t index = 0;
    std::int64_t t0 = 0;
    std::int64_t t1 = 0;
    std::string text;
    bool speaker_turn_next = false;
    float no_speech_prob = 0.0f;
    std::vector<Token> tokens;
};

struct Timings {
    float sample_ms = 0.0f;
    float encode_ms = 0.0f;
    float decode_ms = 0.0f;
    float batchd_ms = 0.0f;
    float prompt_ms = 0.0f;
};

struct TranscribeRequest {
    std::vector<std::uint8_t> audio;
    std::string model_path;

    std::string language = "en";
    bool translate = false;
    bool token_timestamps = false;
    bool single_segment = false;
    bool no_context = true;
    std::int32_t offset_ms = 0;
    std::int32_t duration_ms = 0;
    std::int32_t max_len = 0;
    std::int32_t max_tokens = 0;
    bool split_on_word = false;
    float temperature = 0.0f;
    std::int32_t beam_size = 0;
    std::int32_t best_of = 5;
    std::string initial_prompt;
    bool suppress_blank = true;
    bool suppress_nst = false;
    std::int32_t n_threads = 0;
    bool tdrz_enable = false;
    std::string suppress_regex;

    bool vad_enabled = false;
    std::string vad_model_path;
};

struct TranscribeResponse {
    std::vector<Segment> segments;
    std::string detected_language;
    Timings timings;
};

struct DetectLanguageRequest {
    std::vector<std::uint8_t> audio;
    std::string model_path;
    std::int32_t offset_ms = 0;
    std::int32_t n_threads = 0;
};

struct LanguageProb {
    std::string language;
    float probability = 0.0f;
};

struct DetectLanguageResponse {
    std::string language;
    std::vector<LanguageProb> probs;
};

struct GetModelInfoRequest {
    std::string model_path;
};

struct GetModelInfoResponse {
    std::string model_type;
    bool multilingual = false;
    std::int32_t n_vocab = 0;
    std::int32_t n_audio_ctx = 0;
    std::int32_t n_audio_state = 0;
    std::int32_t n_audio_head = 0;
    std::int32_t n_audio_layer = 0;
    std::int32_t n_text_ctx = 0;
    std::int32_t n_text_state = 0;
    std::int32_t n_text_head = 0;
    std::int32_t n_text_layer = 0;
    std::int32_t n_mels = 0;
    std::int32_t ftype = 0;
};

struct DetectVADRequest {
    std::vector<std::uint8_t> audio;
    std::string vad_model_path;
    float threshold = 0.0f;
    std::int32_t min_speech_duration_ms = 0;
    std::int32_t min_silence_duration_ms = 0;
    float max_speech_duration_s = 0.0f;
    std::int32_t speech_pad_ms = 0;
};

struct VADSegment {
    float t0 = 0.0f;
    float t1 = 0.0f;
};

struct DetectVADResponse {
    bool speech_detected = false;
    std::vector<VADSegment> segments;
};

struct GetVersionResponse {
    std::string version;
    std::string system_info;
};

TranscribeResponse transcribe(const TranscribeRequest& req);
DetectLanguageResponse detect_language(const DetectLanguageRequest& req);
GetModelInfoResponse get_model_info(const GetModelInfoRequest& req);
DetectVADResponse detect_vad(const DetectVADRequest& req);
GetVersionResponse get_version();

}  // namespace whisper_service
