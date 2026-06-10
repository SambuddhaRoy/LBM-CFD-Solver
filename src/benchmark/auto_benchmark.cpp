#include "benchmark/auto_benchmark.h"
#include "benchmark/benchmark_config.h"
#include "benchmark/model_selector.h"
#include "vk_engine.h"
#include "logger.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace vwt::benchmark {

AutoBenchmark::AutoBenchmark(VulkanEngine* engine) : engine_(engine) {}

void AutoBenchmark::run() {
    auto models = ModelSelector::getModels();
    
    std::cout << "\n[BENCHMARK] Starting automated benchmark mode...\n";
    std::cout << "[BENCHMARK] Found " << models.size() << " model(s) to test.\n";

    for (const auto& modelPath : models) {
        std::string modelName = std::filesystem::path(modelPath).filename().string();
        std::cout << "\n[BENCHMARK] Testing model: " << modelName << "\n";

        for (const auto& res : GRID_RESOLUTIONS) {
            // Set resolution
            engine_->baseGridX_ = res.x;
            engine_->baseGridY_ = res.y;
            engine_->baseGridZ_ = res.z;
            engine_->gridQuality_ = 1.0f;
            
            // Trigger simulation re-init (similar to resizePending logic)
            vkDeviceWaitIdle(engine_->device_);
            engine_->simParams_.gridX = res.x;
            engine_->simParams_.gridY = res.y;
            engine_->simParams_.gridZ = res.z;
            engine_->fluidSolver_.destroy();
            engine_->renderer_.destroy();
            engine_->initSimulation();
            engine_->loadMesh(modelPath);

            for (const auto& opName : COLLISION_OPERATORS) {
                engine_->simParams_.lbmMode = (opName == "BGK" ? 0 : 1);

                for (float vel : INLET_VELOCITIES) {
                    engine_->simParams_.inletVelX = vel;
                    engine_->simParams_.inletVelY = 0.0f;
                    engine_->simParams_.inletVelZ = 0.0f;
                    
                    std::string configStr = res.toString() + ", " + opName + ", Vel=" + std::to_string(vel);
                    vwt::Logger::log("  > Starting Config: " + configStr);
                    std::cout << "  > Config: " << configStr << " ... " << std::flush;

                    // Reset simulation state
                    engine_->fluidSolver_.resetToEquilibrium();
                    engine_->totalSteps_ = 0;
                    engine_->simResidual_ = 1.0f;
                    engine_->simRunning_ = true;

                    // 1. Warmup
                    int warmupBatches = WARMUP_STEPS / CHECKPOINT_INTERVAL;
                    for (int w = 0; w < warmupBatches; ++w) {
                        engine_->stepBenchmark(CHECKPOINT_INTERVAL);
                    }
                    // Any remainder
                    if (WARMUP_STEPS % CHECKPOINT_INTERVAL > 0) {
                        engine_->stepBenchmark(WARMUP_STEPS % CHECKPOINT_INTERVAL);
                    }

                    // 2. Measurement
                    int numBatches = MEASUREMENT_STEPS / CHECKPOINT_INTERVAL;
                    for (int b = 0; b < numBatches; ++b) {
                        batchStart_ = std::chrono::steady_clock::now();
                        
                        engine_->stepBenchmark(CHECKPOINT_INTERVAL);
                        std::cout << "." << std::flush;
                        
                        auto now = std::chrono::steady_clock::now();
                        double batchTimeMs = std::chrono::duration<double, std::milli>(now - batchStart_).count();
                        
                        recordMetrics(engine_->totalSteps_, batchTimeMs, modelName, res.toString(), opName, vel);
                    }
                    std::cout << "Done.\n";
                }
            }
        }
    }

    std::cout << "\n[BENCHMARK] All tests completed. Writing results to Excel...\n";
    writeExcel();
    
    ModelSelector::cleanupProceduralModels();

    // Print summary table to stdout
    std::cout << "\n" << std::string(110, '=') << "\n";
    std::cout << std::left << std::setw(20) << "Model" 
              << std::setw(15) << "Grid" 
              << std::setw(10) << "Op" 
              << std::setw(10) << "Vel" 
              << std::right << std::setw(12) << "Avg MLUPS" 
              << std::setw(12) << "Peak MLUPS" 
              << std::setw(10) << "Residual"
              << std::setw(10) << "VRAM GB\n";
    std::cout << std::string(110, '-') << "\n";

    int rid = 0; std::string lcb = "";
    std::vector<BenchmarkMetric> runMetrics;
    
    auto printRun = [&]() {
        if (runMetrics.empty()) return;
        double avgMlups = 0, peakMlups = 0;
        for (const auto& rm : runMetrics) {
            avgMlups += rm.mlups;
            peakMlups = std::max(peakMlups, rm.mlups);
        }
        avgMlups /= runMetrics.size();
        const auto& last = runMetrics.back();
        
        std::cout << std::left << std::setw(20) << (last.modelName.length() > 19 ? last.modelName.substr(0, 16) + "..." : last.modelName)
                  << std::setw(15) << last.gridRes
                  << std::setw(10) << last.operatorName
                  << std::setw(10) << last.inletVelocity
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << avgMlups
                  << std::setw(12) << peakMlups
                  << std::setw(10) << std::scientific << std::setprecision(1) << last.residual
                  << std::fixed << std::setprecision(2) << std::setw(10) << (double)last.vramUsage / (1024.0*1024.0*1024.0) << "\n";
    };

    for (const auto& m : metrics_) {
        std::string cb = m.modelName + m.gridRes + m.operatorName + std::to_string(m.inletVelocity);
        if (cb != lcb) {
            printRun();
            runMetrics.clear();
            lcb = cb;
            rid++;
        }
        runMetrics.push_back(m);
    }
    printRun(); // last one
    
    std::cout << std::string(110, '=') << "\n";
    std::cout << "[BENCHMARK] Benchmark report saved to benchmark_output/VWT_Benchmark_Results.xlsx\n";
    vwt::Logger::log("Benchmark completed successfully.");
}

void AutoBenchmark::recordMetrics(uint64_t step, double batchTimeMs, const std::string& model, 
                                 const std::string& grid, const std::string& op, float vel) {
    BenchmarkMetric m;
    m.step = step;
    m.batchTimeMs = batchTimeMs;
    
    uint64_t numCells = uint64_t(engine_->simParams_.gridX) * engine_->simParams_.gridY * engine_->simParams_.gridZ;
    m.mlups = (double(numCells) * CHECKPOINT_INTERVAL) / (batchTimeMs * 1000.0);
    
    m.lbmMs = engine_->gpuTimings_.lbmMs;
    m.aeroMs = engine_->gpuTimings_.aeroMs;
    m.drag = engine_->aeroForces_.drag;
    m.lift = engine_->aeroForces_.lift;
    m.residual = engine_->simResidual_;
    m.vramUsage = engine_->vramUsage_;
    m.vramBudget = engine_->vramBudget_;
    m.gpuName = engine_->gpuName_;
    m.modelName = model;
    m.gridRes = grid;
    m.operatorName = op;
    m.inletVelocity = vel;
    
    metrics_.push_back(m);
}

void AutoBenchmark::writeExcel() {
    std::string outDir = "benchmark_output";
    if (!std::filesystem::exists(outDir)) {
        std::filesystem::create_directories(outDir);
    }
    std::string outFile = outDir + "/VWT_Benchmark_Results.xlsx";

    lxw_workbook  *workbook  = workbook_new(outFile.c_str());
    
    // Formats
    lxw_format *fmtHeader = workbook_add_format(workbook);
    format_set_bold(fmtHeader);
    format_set_bg_color(fmtHeader, 0xD7E4BC);
    format_set_border(fmtHeader, LXW_BORDER_THIN);

    lxw_format *fmtGreen = workbook_add_format(workbook);
    format_set_bg_color(fmtGreen, 0xC6EFCE);
    format_set_font_color(fmtGreen, 0x006100);

    lxw_format *fmtRed = workbook_add_format(workbook);
    format_set_bg_color(fmtRed, 0xFFC7CE);
    format_set_font_color(fmtRed, 0x9C0006);

    // Sheets
    lxw_worksheet *wsSummary = workbook_add_worksheet(workbook, "Summary");
    lxw_worksheet *wsConv    = workbook_add_worksheet(workbook, "Convergence");
    lxw_worksheet *wsTP      = workbook_add_worksheet(workbook, "Throughput");
    lxw_worksheet *wsAero    = workbook_add_worksheet(workbook, "Aero Forces");
    lxw_worksheet *wsMem     = workbook_add_worksheet(workbook, "Memory");
    lxw_worksheet *wsRaw     = workbook_add_worksheet(workbook, "Raw Data");

    // Summary Header
    const char* summaryHeaders[] = {"Model", "Grid", "Operator", "Inlet Vel", "Avg MLUPS", "Peak MLUPS", "Avg LBM ms", "Final Drag", "Final Lift", "Final Residual", "VRAM Used (GB)", "GPU"};
    for (lxw_col_t i = 0; i < 12; ++i) worksheet_write_string(wsSummary, 0, i, summaryHeaders[i], fmtHeader);

    // Convergence Header
    const char* convHeaders[] = {"Run ID", "Model", "Grid", "Operator", "Inlet Vel", "Step", "Residual (log10)"};
    for (lxw_col_t i = 0; i < 7; ++i) worksheet_write_string(wsConv, 0, i, convHeaders[i], fmtHeader);

    // Throughput Header
    const char* tpHeaders[] = {"Run ID", "Model", "Grid", "Operator", "Inlet Vel", "Step", "MLUPS", "LBM ms", "Aero ms"};
    for (lxw_col_t i = 0; i < 9; ++i) worksheet_write_string(wsTP, 0, i, tpHeaders[i], fmtHeader);

    // Aero Headers
    const char* aeroHeaders[] = {"Run ID", "Model", "Grid", "Operator", "Step", "Raw Drag", "Raw Lift"};
    for (lxw_col_t i = 0; i < 7; ++i) worksheet_write_string(wsAero, 0, i, aeroHeaders[i], fmtHeader);

    // Memory Headers
    const char* memHeaders[] = {"Run ID", "Grid", "Step", "VRAM Used (GB)", "VRAM Budget (GB)", "Usage %"};
    for (lxw_col_t i = 0; i < 6; ++i) worksheet_write_string(wsMem, 0, i, memHeaders[i], fmtHeader);

    // Raw Data Header
    const char* rawHeaders[] = {"Step", "BatchTimeMs", "MLUPS", "LBM ms", "Aero ms", "Drag", "Lift", "Residual", "VRAM Used", "VRAM Budget", "GPU", "Model", "Grid", "Operator", "Velocity"};
    for (lxw_col_t i = 0; i < 15; ++i) worksheet_write_string(wsRaw, 0, i, rawHeaders[i], fmtHeader);

    // Write Data
    uint32_t rowRaw = 1;
    uint32_t rowConv = 1;
    uint32_t rowTP = 1;
    uint32_t rowAero = 1;
    uint32_t rowMem = 1;
    
    // Run ID logic: (model, grid, op, vel) combo
    int currentRunId = 0;
    std::string lastCombo = "";

    for (const auto& m : metrics_) {
        std::string combo = m.modelName + m.gridRes + m.operatorName + std::to_string(m.inletVelocity);
        if (combo != lastCombo) {
            currentRunId++;
            lastCombo = combo;
        }

        // Raw Data
        worksheet_write_number(wsRaw, rowRaw, 0, (double)m.step, NULL);
        worksheet_write_number(wsRaw, rowRaw, 1, m.batchTimeMs, NULL);
        worksheet_write_number(wsRaw, rowRaw, 2, m.mlups, NULL);
        worksheet_write_number(wsRaw, rowRaw, 3, (double)m.lbmMs, NULL);
        worksheet_write_number(wsRaw, rowRaw, 4, (double)m.aeroMs, NULL);
        worksheet_write_number(wsRaw, rowRaw, 5, (double)m.drag, NULL);
        worksheet_write_number(wsRaw, rowRaw, 6, (double)m.lift, NULL);
        worksheet_write_number(wsRaw, rowRaw, 7, (double)m.residual, NULL);
        worksheet_write_number(wsRaw, rowRaw, 8, (double)m.vramUsage, NULL);
        worksheet_write_number(wsRaw, rowRaw, 9, (double)m.vramBudget, NULL);
        worksheet_write_string(wsRaw, rowRaw, 10, m.gpuName.c_str(), NULL);
        worksheet_write_string(wsRaw, rowRaw, 11, m.modelName.c_str(), NULL);
        worksheet_write_string(wsRaw, rowRaw, 12, m.gridRes.c_str(), NULL);
        worksheet_write_string(wsRaw, rowRaw, 13, m.operatorName.c_str(), NULL);
        worksheet_write_number(wsRaw, rowRaw, 14, (double)m.inletVelocity, NULL);
        rowRaw++;

        // Convergence
        worksheet_write_number(wsConv, rowConv, 0, (double)currentRunId, NULL);
        worksheet_write_string(wsConv, rowConv, 1, m.modelName.c_str(), NULL);
        worksheet_write_string(wsConv, rowConv, 2, m.gridRes.c_str(), NULL);
        worksheet_write_string(wsConv, rowConv, 3, m.operatorName.c_str(), NULL);
        worksheet_write_number(wsConv, rowConv, 4, (double)m.inletVelocity, NULL);
        worksheet_write_number(wsConv, rowConv, 5, (double)m.step, NULL);
        worksheet_write_number(wsConv, rowConv, 6, std::log10(std::max(m.residual, 1e-12f)), NULL);
        rowConv++;

        // Throughput
        worksheet_write_number(wsTP, rowTP, 0, (double)currentRunId, NULL);
        worksheet_write_string(wsTP, rowTP, 1, m.modelName.c_str(), NULL);
        worksheet_write_string(wsTP, rowTP, 2, m.gridRes.c_str(), NULL);
        worksheet_write_string(wsTP, rowTP, 3, m.operatorName.c_str(), NULL);
        worksheet_write_number(wsTP, rowTP, 4, (double)m.inletVelocity, NULL);
        worksheet_write_number(wsTP, rowTP, 5, (double)m.step, NULL);
        worksheet_write_number(wsTP, rowTP, 6, m.mlups, NULL);
        worksheet_write_number(wsTP, rowTP, 7, (double)m.lbmMs, NULL);
        worksheet_write_number(wsTP, rowTP, 8, (double)m.aeroMs, NULL);
        rowTP++;

        // Aero
        worksheet_write_number(wsAero, rowAero, 0, (double)currentRunId, NULL);
        worksheet_write_string(wsAero, rowAero, 1, m.modelName.c_str(), NULL);
        worksheet_write_string(wsAero, rowAero, 2, m.gridRes.c_str(), NULL);
        worksheet_write_string(wsAero, rowAero, 3, m.operatorName.c_str(), NULL);
        worksheet_write_number(wsAero, rowAero, 4, (double)m.step, NULL);
        worksheet_write_number(wsAero, rowAero, 5, (double)m.drag, NULL);
        worksheet_write_number(wsAero, rowAero, 6, (double)m.lift, NULL);
        rowAero++;

        // Memory
        worksheet_write_number(wsMem, rowMem, 0, (double)currentRunId, NULL);
        worksheet_write_string(wsMem, rowMem, 1, m.gridRes.c_str(), NULL);
        worksheet_write_number(wsMem, rowMem, 2, (double)m.step, NULL);
        worksheet_write_number(wsMem, rowMem, 3, (double)m.vramUsage / (1024.0*1024.0*1024.0), NULL);
        worksheet_write_number(wsMem, rowMem, 4, (double)m.vramBudget / (1024.0*1024.0*1024.0), NULL);
        worksheet_write_number(wsMem, rowMem, 5, (double)m.vramUsage * 100.0 / (double)m.vramBudget, NULL);
        rowMem++;
    }

    // Write Summary — group metrics by run using the same combo key used above
    {
        std::vector<std::vector<BenchmarkMetric>> runGroups;
        std::string lastCb;
        for (const auto& m : metrics_) {
            std::string cb = m.modelName + m.gridRes + m.operatorName + std::to_string(m.inletVelocity);
            if (cb != lastCb) { runGroups.emplace_back(); lastCb = cb; }
            runGroups.back().push_back(m);
        }

        uint32_t rowSummary = 1;
        for (const auto& runMetrics : runGroups) {
            double avgMlups = 0, peakMlups = 0, avgLbm = 0;
            for (const auto& rm : runMetrics) {
                avgMlups += rm.mlups;
                peakMlups = std::max(peakMlups, rm.mlups);
                avgLbm += rm.lbmMs;
            }
            avgMlups /= runMetrics.size();
            avgLbm   /= runMetrics.size();

            const auto& last = runMetrics.back();
            worksheet_write_string(wsSummary, rowSummary, 0, last.modelName.c_str(), NULL);
            worksheet_write_string(wsSummary, rowSummary, 1, last.gridRes.c_str(), NULL);
            worksheet_write_string(wsSummary, rowSummary, 2, last.operatorName.c_str(), NULL);
            worksheet_write_number(wsSummary, rowSummary, 3, (double)last.inletVelocity, NULL);
            worksheet_write_number(wsSummary, rowSummary, 4, avgMlups, NULL);
            worksheet_write_number(wsSummary, rowSummary, 5, peakMlups, NULL);
            worksheet_write_number(wsSummary, rowSummary, 6, (double)avgLbm, NULL);
            worksheet_write_number(wsSummary, rowSummary, 7, (double)last.drag, NULL);
            worksheet_write_number(wsSummary, rowSummary, 8, (double)last.lift, NULL);
            worksheet_write_number(wsSummary, rowSummary, 9, std::log10(std::max(last.residual, 1e-12f)), NULL);
            worksheet_write_number(wsSummary, rowSummary, 10, (double)last.vramUsage / (1024.0*1024.0*1024.0), NULL);
            worksheet_write_string(wsSummary, rowSummary, 11, last.gpuName.c_str(), NULL);

            lxw_format* rowFmt = (avgMlups > 3000.0) ? fmtGreen : NULL;
            (void)rowFmt;
            rowSummary++;
        }

        // Conditional formatting on Avg MLUPS column
        lxw_conditional_format *condGreen = (lxw_conditional_format*)calloc(1, sizeof(lxw_conditional_format));
        condGreen->type = LXW_CONDITIONAL_TYPE_TOP;
        condGreen->value = 10;
        condGreen->format = fmtGreen;
        worksheet_conditional_format_range(wsSummary, 1, 4, rowSummary-1, 4, condGreen);

        lxw_conditional_format *condRed = (lxw_conditional_format*)calloc(1, sizeof(lxw_conditional_format));
        condRed->type = LXW_CONDITIONAL_TYPE_BOTTOM;
        condRed->value = 10;
        condRed->format = fmtRed;
        worksheet_conditional_format_range(wsSummary, 1, 4, rowSummary-1, 4, condRed);

        free(condGreen);
        free(condRed);
    }

    workbook_close(workbook);
}

} // namespace vwt::benchmark
