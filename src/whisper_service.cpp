#include "whisper_service.h"

#include "whisper.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace whisper_service {
namespace {

struct WhisperContextDeleter {
    void operator()(whisper_context* ctx) const {
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
    }
};

struct WhisperVADContextDeleter {
    void operator()(whisper_vad_context* ctx) const {
        if (ctx != nullptr) {
            whisper_vad_free(ctx);
        }
    }
};

struct WhisperVADSegmentsDeleter {
    void operator()(whisper_vad_segments* segments) const {
        if (segments != nullptr) {
            whisper_vad_free_segments(segments);
        }
    }
};

using WhisperContextPtr = std::shared_ptr<whisper_context>;
using WhisperVADContextPtr = std::unique_ptr<whisper_vad_context, WhisperVADContextDeleter>;
using WhisperVADSegmentsPtr = std::unique_ptr<whisper_vad_segments, WhisperVADSegmentsDeleter>;

struct CachedModelContexts {
    WhisperContextPtr regular;
    WhisperContextPtr dtw;
};

std::unordered_map<std::string, CachedModelContexts> g_model_cache;
std::mutex g_model_cache_mutex;

int default_threads() {
    const unsigned int n = std::thread::hardware_concurrency();
    return static_cast<int>(n == 0 ? 1U : n);
}

std::vector<float> decode_pcm_f32(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty()) {
        return {};
    }
    if ((bytes.size() % sizeof(float)) != 0U) {
        throw std::invalid_argument("audio byte length must be a multiple of 4 for float32 PCM");
    }

    std::vector<float> pcm(bytes.size() / sizeof(float));
    std::memcpy(pcm.data(), bytes.data(), bytes.size());
    return pcm;
}

WhisperContextPtr load_context(const std::string& model_path, bool enable_dtw) {
    whisper_context_params cparams = whisper_context_default_params();
    if (enable_dtw) {
        // DTW alignment requires explicit head selection and disabling flash attention.
        cparams.flash_attn = false;
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_N_TOP_MOST;
        cparams.dtw_n_top = 4;
    }

    whisper_context* raw = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (raw == nullptr) {
        throw std::runtime_error("failed to load whisper model: " + model_path);
    }
    return WhisperContextPtr(raw, WhisperContextDeleter{});
}

WhisperContextPtr get_or_load_context(const std::string& model_path, bool enable_dtw) {
    if (model_path.empty()) {
        throw std::invalid_argument("model_path is required");
    }

    {
        std::lock_guard<std::mutex> lock(g_model_cache_mutex);
        auto it = g_model_cache.find(model_path);
        if (it != g_model_cache.end()) {
            auto& cached = it->second;
            if (enable_dtw && cached.dtw) {
                return cached.dtw;
            }
            if (!enable_dtw && cached.regular) {
                return cached.regular;
            }
        }
    }

    auto loaded = load_context(model_path, enable_dtw);

    std::lock_guard<std::mutex> lock(g_model_cache_mutex);
    auto& slot = g_model_cache[model_path];
    auto& target = enable_dtw ? slot.dtw : slot.regular;
    if (!target) {
        target = loaded;
    }
    return target;
}

}  // namespace

TranscribeResponse transcribe(const TranscribeRequest& req) {
    TranscribeResponse res;

    if (req.audio.empty()) {
        return res;
    }

    const std::vector<float> pcm = decode_pcm_f32(req.audio);
    auto ctx = get_or_load_context(req.model_path, req.token_timestamps);

    whisper_sampling_strategy strategy = req.beam_size > 0 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
    whisper_full_params params = whisper_full_default_params(strategy);

    params.n_threads = req.n_threads > 0 ? req.n_threads : default_threads();
    params.translate = req.translate;
    params.token_timestamps = req.token_timestamps;
    params.single_segment = req.single_segment;
    params.no_context = req.no_context;
    params.offset_ms = req.offset_ms;
    params.duration_ms = req.duration_ms;
    params.max_len = req.max_len;
    params.max_tokens = req.max_tokens;
    params.split_on_word = req.split_on_word;
    params.temperature = req.temperature;
    params.suppress_blank = req.suppress_blank;
    params.suppress_nst = req.suppress_nst;
    params.tdrz_enable = req.tdrz_enable;

    if (req.beam_size > 0) {
        params.beam_search.beam_size = req.beam_size;
    } else if (req.best_of > 0) {
        params.greedy.best_of = req.best_of;
    }

    if (!req.initial_prompt.empty()) {
        params.initial_prompt = req.initial_prompt.c_str();
    }
    if (!req.suppress_regex.empty()) {
        params.suppress_regex = req.suppress_regex.c_str();
    }

    if (req.language.empty() || req.language == "auto") {
        params.language = nullptr;
        params.detect_language = true;
    } else {
        params.language = req.language.c_str();
        params.detect_language = false;
    }

    if (req.vad_enabled) {
        params.vad = true;
        if (!req.vad_model_path.empty()) {
            params.vad_model_path = req.vad_model_path.c_str();
        }
    }

    if (whisper_full(ctx.get(), params, pcm.data(), static_cast<int>(pcm.size())) != 0) {
        throw std::runtime_error("whisper_full failed");
    }

    const int lang_id = whisper_full_lang_id(ctx.get());
    if (lang_id >= 0) {
        const char* lang = whisper_lang_str(lang_id);
        if (lang != nullptr) {
            res.detected_language = lang;
        }
    }

    const int n_segments = whisper_full_n_segments(ctx.get());
    res.segments.reserve(std::max(0, n_segments));

    for (int i = 0; i < n_segments; ++i) {
        Segment segment;
        segment.index = i;
        segment.t0 = whisper_full_get_segment_t0(ctx.get(), i);
        segment.t1 = whisper_full_get_segment_t1(ctx.get(), i);

        if (const char* text = whisper_full_get_segment_text(ctx.get(), i); text != nullptr) {
            segment.text = text;
        }

        segment.speaker_turn_next = whisper_full_get_segment_speaker_turn_next(ctx.get(), i);
        segment.no_speech_prob = whisper_full_get_segment_no_speech_prob(ctx.get(), i);

        const int n_tokens = whisper_full_n_tokens(ctx.get(), i);
        segment.tokens.reserve(std::max(0, n_tokens));

        for (int j = 0; j < n_tokens; ++j) {
            const whisper_token_data data = whisper_full_get_token_data(ctx.get(), i, j);

            Token token;
            token.id = data.id;
            if (const char* token_text = whisper_full_get_token_text(ctx.get(), i, j); token_text != nullptr) {
                token.text = token_text;
            }
            token.p = data.p;
            token.plog = data.plog;
            token.t0 = data.t0;
            token.t1 = data.t1;
            token.t_dtw = data.t_dtw;
            token.vlen = data.vlen;

            segment.tokens.push_back(std::move(token));
        }

        res.segments.push_back(std::move(segment));
    }

    if (whisper_timings* timings = whisper_get_timings(ctx.get()); timings != nullptr) {
        res.timings.sample_ms = timings->sample_ms;
        res.timings.encode_ms = timings->encode_ms;
        res.timings.decode_ms = timings->decode_ms;
        res.timings.batchd_ms = timings->batchd_ms;
        res.timings.prompt_ms = timings->prompt_ms;
    }

    return res;
}

DetectLanguageResponse detect_language(const DetectLanguageRequest& req) {
    DetectLanguageResponse res;

    if (req.audio.empty()) {
        return res;
    }

    const std::vector<float> pcm = decode_pcm_f32(req.audio);
    auto ctx = get_or_load_context(req.model_path, false);

    const int n_threads = req.n_threads > 0 ? req.n_threads : default_threads();

    if (whisper_pcm_to_mel(ctx.get(), pcm.data(), static_cast<int>(pcm.size()), n_threads) != 0) {
        throw std::runtime_error("whisper_pcm_to_mel failed");
    }

    const int max_lang_id = whisper_lang_max_id();
    std::vector<float> probs(static_cast<std::size_t>(max_lang_id + 1), 0.0f);

    const int lang_id = whisper_lang_auto_detect(ctx.get(), req.offset_ms, n_threads, probs.data());
    if (lang_id < 0) {
        throw std::runtime_error("whisper_lang_auto_detect failed");
    }

    if (const char* lang = whisper_lang_str(lang_id); lang != nullptr) {
        res.language = lang;
    }

    res.probs.reserve(probs.size());
    for (int i = 0; i <= max_lang_id; ++i) {
        const char* lang = whisper_lang_str(i);
        if (lang == nullptr) {
            continue;
        }
        res.probs.push_back(LanguageProb{lang, probs[static_cast<std::size_t>(i)]});
    }

    return res;
}

GetModelInfoResponse get_model_info(const GetModelInfoRequest& req) {
    auto ctx = get_or_load_context(req.model_path, false);

    GetModelInfoResponse res;
    if (const char* model_type = whisper_model_type_readable(ctx.get()); model_type != nullptr) {
        res.model_type = model_type;
    } else {
        res.model_type = std::to_string(whisper_model_type(ctx.get()));
    }

    res.multilingual = whisper_is_multilingual(ctx.get()) != 0;
    res.n_vocab = whisper_model_n_vocab(ctx.get());
    res.n_audio_ctx = whisper_model_n_audio_ctx(ctx.get());
    res.n_audio_state = whisper_model_n_audio_state(ctx.get());
    res.n_audio_head = whisper_model_n_audio_head(ctx.get());
    res.n_audio_layer = whisper_model_n_audio_layer(ctx.get());
    res.n_text_ctx = whisper_model_n_text_ctx(ctx.get());
    res.n_text_state = whisper_model_n_text_state(ctx.get());
    res.n_text_head = whisper_model_n_text_head(ctx.get());
    res.n_text_layer = whisper_model_n_text_layer(ctx.get());
    res.n_mels = whisper_model_n_mels(ctx.get());
    res.ftype = whisper_model_ftype(ctx.get());
    return res;
}

DetectVADResponse detect_vad(const DetectVADRequest& req) {
    DetectVADResponse res;

    if (req.audio.empty()) {
        return res;
    }
    if (req.vad_model_path.empty()) {
        throw std::invalid_argument("vad_model_path is required");
    }

    const std::vector<float> pcm = decode_pcm_f32(req.audio);

    whisper_vad_context_params cparams = whisper_vad_default_context_params();
    whisper_vad_context* raw_vctx = whisper_vad_init_from_file_with_params(req.vad_model_path.c_str(), cparams);
    if (raw_vctx == nullptr) {
        throw std::runtime_error("failed to initialize VAD model: " + req.vad_model_path);
    }
    WhisperVADContextPtr vctx(raw_vctx);

    whisper_vad_params params = whisper_vad_default_params();
    if (req.threshold > 0.0f) {
        params.threshold = req.threshold;
    }
    if (req.min_speech_duration_ms > 0) {
        params.min_speech_duration_ms = req.min_speech_duration_ms;
    }
    if (req.min_silence_duration_ms > 0) {
        params.min_silence_duration_ms = req.min_silence_duration_ms;
    }
    if (req.max_speech_duration_s > 0.0f) {
        params.max_speech_duration_s = req.max_speech_duration_s;
    }
    if (req.speech_pad_ms > 0) {
        params.speech_pad_ms = req.speech_pad_ms;
    }

    const bool detected = whisper_vad_detect_speech(vctx.get(), pcm.data(), static_cast<int>(pcm.size()));

    whisper_vad_segments* raw_segments = whisper_vad_segments_from_samples(
        vctx.get(), params, pcm.data(), static_cast<int>(pcm.size()));
    WhisperVADSegmentsPtr segments(raw_segments);

    if (segments != nullptr) {
        const int n_segments = whisper_vad_segments_n_segments(segments.get());
        res.segments.reserve(std::max(0, n_segments));

        for (int i = 0; i < n_segments; ++i) {
            VADSegment segment;
            segment.t0 = whisper_vad_segments_get_segment_t0(segments.get(), i);
            segment.t1 = whisper_vad_segments_get_segment_t1(segments.get(), i);
            res.segments.push_back(segment);
        }
    }

    res.speech_detected = detected || !res.segments.empty();
    return res;
}

GetVersionResponse get_version() {
    GetVersionResponse res;

    if (const char* version = whisper_version(); version != nullptr) {
        res.version = version;
    }
    if (const char* info = whisper_print_system_info(); info != nullptr) {
        res.system_info = info;
    }

    return res;
}

}  // namespace whisper_service
