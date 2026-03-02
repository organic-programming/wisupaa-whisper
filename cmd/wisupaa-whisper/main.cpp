#include "whisper_service.h"

#include "holons/holons.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;

namespace {

std::vector<std::uint8_t> base64_decode(const std::string& input) {
    static const int kTable[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,63,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };

    std::vector<std::uint8_t> out;
    int val = 0;
    int valb = -8;
    for (unsigned char c : input) {
        if (c == '=') {
            break;
        }
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            continue;
        }
        int d = kTable[c];
        if (d == -1) {
            throw std::invalid_argument("invalid base64 in bytes field");
        }
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<std::uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::vector<std::uint8_t> parse_bytes_field(const json& obj, const char* key) {
    if (!obj.contains(key) || obj[key].is_null()) {
        return {};
    }

    const json& field = obj[key];
    if (field.is_string()) {
        return base64_decode(field.get<std::string>());
    }

    if (field.is_array()) {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(field.size());
        for (const auto& v : field) {
            if (!v.is_number_integer()) {
                throw std::invalid_argument(std::string("bytes field '") + key + "' must contain integers");
            }
            int value = v.get<int>();
            if (value < 0 || value > 255) {
                throw std::invalid_argument(std::string("bytes field '") + key + "' contains out-of-range byte");
            }
            bytes.push_back(static_cast<std::uint8_t>(value));
        }
        return bytes;
    }

    throw std::invalid_argument(std::string("bytes field '") + key + "' must be base64 string or byte array");
}

bool write_all(const holons::connection& conn, std::string_view data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        ssize_t n = holons::conn_write(conn, data.data() + offset, data.size() - offset);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        offset += static_cast<std::size_t>(n);
    }
    return true;
}

json make_error(const json& id, int code, const std::string& message) {
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", message},
        }}
    };
}

json token_to_json(const whisper_service::Token& token) {
    return {
        {"id", token.id},
        {"text", token.text},
        {"p", token.p},
        {"plog", token.plog},
        {"t0", token.t0},
        {"t1", token.t1},
        {"t_dtw", token.t_dtw},
        {"vlen", token.vlen},
    };
}

json segment_to_json(const whisper_service::Segment& segment) {
    json tokens = json::array();
    for (const auto& token : segment.tokens) {
        tokens.push_back(token_to_json(token));
    }

    return {
        {"index", segment.index},
        {"t0", segment.t0},
        {"t1", segment.t1},
        {"text", segment.text},
        {"speaker_turn_next", segment.speaker_turn_next},
        {"no_speech_prob", segment.no_speech_prob},
        {"tokens", tokens},
    };
}

json timings_to_json(const whisper_service::Timings& timings) {
    return {
        {"sample_ms", timings.sample_ms},
        {"encode_ms", timings.encode_ms},
        {"decode_ms", timings.decode_ms},
        {"batchd_ms", timings.batchd_ms},
        {"prompt_ms", timings.prompt_ms},
    };
}

whisper_service::TranscribeRequest parse_transcribe_request(const json& params) {
    whisper_service::TranscribeRequest req;

    req.audio = parse_bytes_field(params, "audio");
    req.model_path = params.value("model_path", std::string{});
    req.language = params.value("language", req.language);
    req.translate = params.value("translate", req.translate);
    req.token_timestamps = params.value("token_timestamps", req.token_timestamps);
    req.single_segment = params.value("single_segment", req.single_segment);
    req.no_context = params.value("no_context", req.no_context);
    req.offset_ms = params.value("offset_ms", req.offset_ms);
    req.duration_ms = params.value("duration_ms", req.duration_ms);
    req.max_len = params.value("max_len", req.max_len);
    req.max_tokens = params.value("max_tokens", req.max_tokens);
    req.split_on_word = params.value("split_on_word", req.split_on_word);
    req.temperature = params.value("temperature", req.temperature);
    req.beam_size = params.value("beam_size", req.beam_size);
    req.best_of = params.value("best_of", req.best_of);
    req.initial_prompt = params.value("initial_prompt", req.initial_prompt);
    req.suppress_blank = params.value("suppress_blank", req.suppress_blank);
    req.suppress_nst = params.value("suppress_nst", req.suppress_nst);
    req.n_threads = params.value("n_threads", req.n_threads);
    req.tdrz_enable = params.value("tdrz_enable", req.tdrz_enable);
    req.suppress_regex = params.value("suppress_regex", req.suppress_regex);
    req.vad_enabled = params.value("vad_enabled", req.vad_enabled);
    req.vad_model_path = params.value("vad_model_path", req.vad_model_path);

    return req;
}

whisper_service::DetectLanguageRequest parse_detect_language_request(const json& params) {
    whisper_service::DetectLanguageRequest req;
    req.audio = parse_bytes_field(params, "audio");
    req.model_path = params.value("model_path", std::string{});
    req.offset_ms = params.value("offset_ms", req.offset_ms);
    req.n_threads = params.value("n_threads", req.n_threads);
    return req;
}

whisper_service::GetModelInfoRequest parse_get_model_info_request(const json& params) {
    whisper_service::GetModelInfoRequest req;
    req.model_path = params.value("model_path", std::string{});
    return req;
}

whisper_service::DetectVADRequest parse_detect_vad_request(const json& params) {
    whisper_service::DetectVADRequest req;
    req.audio = parse_bytes_field(params, "audio");
    req.vad_model_path = params.value("vad_model_path", std::string{});
    req.threshold = params.value("threshold", req.threshold);
    req.min_speech_duration_ms = params.value("min_speech_duration_ms", req.min_speech_duration_ms);
    req.min_silence_duration_ms = params.value("min_silence_duration_ms", req.min_silence_duration_ms);
    req.max_speech_duration_s = params.value("max_speech_duration_s", req.max_speech_duration_s);
    req.speech_pad_ms = params.value("speech_pad_ms", req.speech_pad_ms);
    return req;
}

json handle_request(const json& request) {
    const json id = request.contains("id") ? request["id"] : json(nullptr);

    if (!request.is_object()) {
        return make_error(id, -32600, "Invalid Request");
    }
    if (!request.contains("jsonrpc") || request["jsonrpc"] != "2.0") {
        return make_error(id, -32600, "Invalid Request");
    }
    if (!request.contains("method") || !request["method"].is_string()) {
        return make_error(id, -32600, "Invalid Request");
    }

    const std::string method = request["method"].get<std::string>();
    json params = json::object();
    if (request.contains("params")) {
        params = request["params"];
        if (!params.is_object()) {
            return make_error(id, -32602, "Invalid params");
        }
    }

    try {
        if (method == "transcribe") {
            const auto req = parse_transcribe_request(params);
            const auto res = whisper_service::transcribe(req);

            json segments = json::array();
            for (const auto& segment : res.segments) {
                segments.push_back(segment_to_json(segment));
            }

            return {
                {"jsonrpc", "2.0"},
                {"id", id},
                {"result", {
                    {"segments", segments},
                    {"detected_language", res.detected_language},
                    {"timings", timings_to_json(res.timings)},
                }}
            };
        }

        if (method == "detect_language") {
            const auto req = parse_detect_language_request(params);
            const auto res = whisper_service::detect_language(req);

            json probs = json::array();
            for (const auto& p : res.probs) {
                probs.push_back({
                    {"language", p.language},
                    {"probability", p.probability},
                });
            }

            return {
                {"jsonrpc", "2.0"},
                {"id", id},
                {"result", {
                    {"language", res.language},
                    {"probs", probs},
                }}
            };
        }

        if (method == "get_model_info") {
            const auto req = parse_get_model_info_request(params);
            const auto res = whisper_service::get_model_info(req);

            return {
                {"jsonrpc", "2.0"},
                {"id", id},
                {"result", {
                    {"model_type", res.model_type},
                    {"multilingual", res.multilingual},
                    {"n_vocab", res.n_vocab},
                    {"n_audio_ctx", res.n_audio_ctx},
                    {"n_audio_state", res.n_audio_state},
                    {"n_audio_head", res.n_audio_head},
                    {"n_audio_layer", res.n_audio_layer},
                    {"n_text_ctx", res.n_text_ctx},
                    {"n_text_state", res.n_text_state},
                    {"n_text_head", res.n_text_head},
                    {"n_text_layer", res.n_text_layer},
                    {"n_mels", res.n_mels},
                    {"ftype", res.ftype},
                }}
            };
        }

        if (method == "detect_vad") {
            const auto req = parse_detect_vad_request(params);
            const auto res = whisper_service::detect_vad(req);

            json segments = json::array();
            for (const auto& segment : res.segments) {
                segments.push_back({
                    {"t0", segment.t0},
                    {"t1", segment.t1},
                });
            }

            return {
                {"jsonrpc", "2.0"},
                {"id", id},
                {"result", {
                    {"speech_detected", res.speech_detected},
                    {"segments", segments},
                }}
            };
        }

        if (method == "get_version") {
            const auto res = whisper_service::get_version();
            return {
                {"jsonrpc", "2.0"},
                {"id", id},
                {"result", {
                    {"version", res.version},
                    {"system_info", res.system_info},
                }}
            };
        }

        return make_error(id, -32601, "Method not found");
    } catch (const std::invalid_argument& e) {
        return make_error(id, -32602, e.what());
    } catch (const std::exception& e) {
        return make_error(id, -32000, e.what());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        auto identity = holons::parse_holon("HOLON.md");
        std::cerr << "[" << identity.given_name << " " << identity.family_name << "] "
                  << identity.motto << std::endl;

        std::vector<std::string> args(argv + 1, argv + argc);
        const bool has_explicit_transport =
            std::find(args.begin(), args.end(), "--listen") != args.end() ||
            std::find(args.begin(), args.end(), "--port") != args.end();
        const std::string uri = has_explicit_transport ? holons::parse_flags(args) : "stdio://";

        auto lis = holons::listen(uri);
        auto conn = holons::accept(lis);

        std::string pending;
        std::vector<char> buf(8192);

        while (true) {
            const ssize_t n = holons::conn_read(conn, buf.data(), buf.size());
            if (n == 0) {
                break;
            }
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
            }

            pending.append(buf.data(), static_cast<std::size_t>(n));

            std::size_t pos = 0;
            while (true) {
                const std::size_t newline = pending.find('\n', pos);
                if (newline == std::string::npos) {
                    pending.erase(0, pos);
                    break;
                }

                std::string line = pending.substr(pos, newline - pos);
                pos = newline + 1;

                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (line.empty()) {
                    continue;
                }

                json response;
                try {
                    const json request = json::parse(line);
                    response = handle_request(request);
                } catch (const json::parse_error& e) {
                    response = make_error(nullptr, -32700, e.what());
                }

                const std::string payload = response.dump() + "\n";
                if (!write_all(conn, payload)) {
                    break;
                }
            }
        }

        holons::close_connection(conn);
        holons::close_listener(lis);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        return 1;
    }
}
