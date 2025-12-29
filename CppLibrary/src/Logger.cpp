//
// Logger - Logging system for aiRunner
//

#include "../include/Logger.h"
#include <iostream>

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() = default;

Logger::~Logger() {
    closeLogFile();
}

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_level_ = level;
}

LogLevel Logger::getLogLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_level_;
}

void Logger::enableConsole(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    console_enabled_ = enable;
}

void Logger::enableTimestamp(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    timestamp_enabled_ = enable;
}

bool Logger::setLogFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Close existing file if open
    if (file_stream_.is_open()) {
        file_stream_.close();
    }

    file_stream_.open(filepath, std::ios::out | std::ios::app);
    if (file_stream_.is_open()) {
        file_path_ = filepath;
        return true;
    }

    return false;
}

void Logger::closeLogFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    file_path_.clear();
}

void Logger::setCallback(LogCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = callback;
}

void Logger::clearCallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = nullptr;
}

void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we should log this level
    if (level < current_level_) {
        return;
    }

    // Build log message
    std::ostringstream formatted;

    if (timestamp_enabled_) {
        formatted << "[" << getTimestamp() << "] ";
    }

    formatted << "[" << levelToString(level) << "] " << message;

    std::string log_line = formatted.str();

    // Output to console
    if (console_enabled_) {
        if (level == LogLevel::ERROR) {
            std::cerr << log_line << std::endl;
        } else {
            std::cout << log_line << std::endl;
        }
    }

    // Output to file
    if (file_stream_.is_open()) {
        file_stream_ << log_line << std::endl;
        file_stream_.flush();
    }

    // Call custom callback
    if (callback_) {
        callback_(level, message);
    }
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warn(const std::string& message) {
    log(LogLevel::WARN, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        default:              return "UNKNOWN";
    }
}

std::string Logger::getTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}
