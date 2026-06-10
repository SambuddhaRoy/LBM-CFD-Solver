#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace vwt {

class Logger {
public:
    static void init(const std::string& filename = "vwt_session.log") {
        instance().logFile_.open(filename, std::ios::app);
        log("--- Session Started ---");
    }

    static void log(const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        // Thread-safe, cross-platform local time conversion.
        std::tm tmBuf{};
#ifdef _WIN32
        localtime_s(&tmBuf, &time);
#else
        localtime_r(&time, &tmBuf);
#endif

        std::stringstream ss;
        ss << "[" << std::put_time(&tmBuf, "%H:%M:%S")
           << "." << std::setfill('0') << std::setw(3) << ms.count() << "] " 
           << message;

        std::string formatted = ss.str();
        std::cout << formatted << std::endl;
        if (instance().logFile_.is_open()) {
            instance().logFile_ << formatted << std::endl;
        }
    }

    static void error(const std::string& message) {
        log("[ERROR] " + message);
    }

    static void warn(const std::string& message) {
        log("[WARN] " + message);
    }

private:
    Logger() = default;
    static Logger& instance() {
        static Logger inst;
        return inst;
    }
    std::ofstream logFile_;
};

} // namespace vwt
