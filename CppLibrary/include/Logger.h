//
// Logger - Logging system for aiRunner
//

#ifndef AIRUNNER_LOGGER_H
#define AIRUNNER_LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <functional>
#include <sstream>
#include <chrono>
#include <iomanip>

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    NONE = 4  // Disable all logging
};

// Callback type for custom log handling
using LogCallback = std::function<void(LogLevel level, const std::string& message)>;

// Raw C callback (level as int, message as C string, opaque user data).
// Declared here so the public C ABI bridge can register without aiRunner.cpp
// owning any file-scope state.
using CLogCallback = void(*)(int level, const char* message, void* user_data);

class Logger {
public:
    static Logger& getInstance();

    // Configuration
    void setLogLevel(LogLevel level);
    LogLevel getLogLevel() const;

    void enableConsole(bool enable);
    void enableTimestamp(bool enable);

    // File logging
    bool setLogFile(const std::string& filepath);
    void closeLogFile();

    // Custom callback (C++ std::function)
    void setCallback(LogCallback callback);
    void clearCallback();

    // Custom callback (raw C function pointer + opaque user data)
    void setCCallback(CLogCallback callback, void* user_data);
    void clearCCallback();

    // Logging methods
    void log(LogLevel level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);

    // Stream-style logging
    template<typename... Args>
    void debug(Args&&... args) { logFormatted(LogLevel::DEBUG, std::forward<Args>(args)...); }

    template<typename... Args>
    void info(Args&&... args) { logFormatted(LogLevel::INFO, std::forward<Args>(args)...); }

    template<typename... Args>
    void warn(Args&&... args) { logFormatted(LogLevel::WARN, std::forward<Args>(args)...); }

    template<typename... Args>
    void error(Args&&... args) { logFormatted(LogLevel::ERROR, std::forward<Args>(args)...); }

private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string levelToString(LogLevel level) const;
    std::string getTimestamp() const;

    template<typename... Args>
    void logFormatted(LogLevel level, Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        log(level, oss.str());
    }

    LogLevel current_level_ = LogLevel::INFO;
    bool console_enabled_ = true;
    bool timestamp_enabled_ = true;

    std::ofstream file_stream_;
    std::string file_path_;

    LogCallback callback_ = nullptr;

    // C ABI callback bridge — owned here so the public C wrapper has no
    // file-scope statics for log routing.
    CLogCallback c_callback_ = nullptr;
    void* c_callback_user_data_ = nullptr;

    mutable std::mutex mutex_;
};

// Convenience macros
#define LOG_DEBUG(msg) Logger::getInstance().debug(msg)
#define LOG_INFO(msg) Logger::getInstance().info(msg)
#define LOG_WARN(msg) Logger::getInstance().warn(msg)
#define LOG_ERROR(msg) Logger::getInstance().error(msg)

#endif //AIRUNNER_LOGGER_H
