cmake_minimum_required(VERSION 3.14)
project(wisupaa-whisper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# whisper.cpp engine
add_subdirectory(third_party/whisper.cpp)

# cpp-holons SDK — header-only, lives in parent monorepo at
# organic-programming/sdk/cpp-holons/ (relative to holons/wisupaa-whisper/)
set(CPP_HOLONS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../organic-programming/sdk/cpp-holons/include)

# nlohmann/json — the SDK needs it, find it portably.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS /opt/homebrew/include /usr/local/include
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if (NOT EXISTS ${CPP_HOLONS_INCLUDE_DIR}/holons/holons.hpp)
    message(FATAL_ERROR "cpp-holons SDK not found at ${CPP_HOLONS_INCLUDE_DIR}")
endif()

# Service library
add_library(whisper_service src/whisper_service.cpp)
target_include_directories(whisper_service PUBLIC src/)
target_include_directories(whisper_service PRIVATE third_party/whisper.cpp/include)
target_link_libraries(whisper_service PRIVATE whisper)

# CLI executable
add_executable(wisupaa-whisper cmd/wisupaa-whisper/main.cpp)
target_include_directories(wisupaa-whisper PRIVATE ${CPP_HOLONS_INCLUDE_DIR})
target_include_directories(wisupaa-whisper PRIVATE third_party/whisper.cpp/include)
if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(wisupaa-whisper PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif()
target_link_libraries(wisupaa-whisper PRIVATE whisper_service whisper)

# Tests
add_executable(test_whisper_service tests/test_whisper_service.cpp)
target_include_directories(test_whisper_service PRIVATE ${CPP_HOLONS_INCLUDE_DIR})
target_include_directories(test_whisper_service PRIVATE third_party/whisper.cpp/include)
if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(test_whisper_service PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif()
target_link_libraries(test_whisper_service PRIVATE whisper_service whisper)
add_test(NAME whisper_service COMMAND test_whisper_service)
set_tests_properties(whisper_service PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Link Winsock on Windows (required by cpp-holons SDK).
if(WIN32)
    target_link_libraries(wisupaa-whisper PRIVATE ws2_32)
    target_link_libraries(test_whisper_service PRIVATE ws2_32)
endif()
