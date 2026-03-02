#include "whisper_service.h"

#include "holons/holons.hpp"

#include <cstdio>
#include <filesystem>
#include <iostream>

#define ASSERT(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while(0)

namespace {

std::filesystem::path find_holon_path() {
    const std::filesystem::path candidates[] = {
        "HOLON.md",
        "../HOLON.md",
        "../../HOLON.md",
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return "HOLON.md";
}

}  // namespace

int main() {
    whisper_service::Token token;
    whisper_service::Segment segment;
    whisper_service::Timings timings;
    whisper_service::TranscribeRequest transcribe_req;
    whisper_service::TranscribeResponse transcribe_res;
    whisper_service::DetectLanguageRequest detect_lang_req;
    whisper_service::LanguageProb lang_prob;
    whisper_service::DetectLanguageResponse detect_lang_res;
    whisper_service::GetModelInfoRequest model_info_req;
    whisper_service::GetModelInfoResponse model_info_res;
    whisper_service::DetectVADRequest vad_req;
    whisper_service::VADSegment vad_segment;
    whisper_service::DetectVADResponse vad_res;
    whisper_service::GetVersionResponse version_res;

    ASSERT(token.id == 0);
    ASSERT(segment.index == 0);
    ASSERT(timings.sample_ms == 0.0f);
    ASSERT(transcribe_req.no_context == true);
    ASSERT(transcribe_res.segments.empty());
    ASSERT(detect_lang_req.offset_ms == 0);
    ASSERT(lang_prob.probability == 0.0f);
    ASSERT(detect_lang_res.probs.empty());
    ASSERT(model_info_req.model_path.empty());
    ASSERT(model_info_res.n_vocab == 0);
    ASSERT(vad_req.vad_model_path.empty());
    ASSERT(vad_segment.t0 == 0.0f);
    ASSERT(vad_res.speech_detected == false);
    ASSERT(version_res.version.empty());

    const auto runtime_version = whisper_service::get_version();
    ASSERT(!runtime_version.version.empty());

    const whisper_service::TranscribeResponse empty_transcribe = whisper_service::transcribe({});
    ASSERT(empty_transcribe.segments.empty());

    const whisper_service::DetectVADResponse empty_vad = whisper_service::detect_vad({});
    ASSERT(empty_vad.speech_detected == false);
    ASSERT(empty_vad.segments.empty());

    const auto holon = holons::parse_holon(find_holon_path().string());
    ASSERT(!holon.uuid.empty());
    ASSERT(holon.given_name == "Wisupaa");
    ASSERT(holon.family_name == "Whisper");

    std::cout << "All tests passed" << std::endl;
    return 0;
}
