#include "handeye_eval.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage(const char* exeName)
{
    std::cout << "Usage:\n"
        << "  " << exeName << " [dataset_root]\n"
        << "  " << exeName << " --self-test [dataset_root]\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        bool selfTest = false;
        std::filesystem::path root = std::filesystem::current_path();

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (arg == "--self-test") {
                selfTest = true;
                continue;
            }
            root = std::filesystem::path(arg);
        }

        HandEyeEvaluator evaluator;
        if (selfTest) {
            return evaluator.selfTest(root, std::cout) ? 0 : 1;
        }

        const EvaluationReport report = evaluator.evaluate(root);
        const std::filesystem::path reportPath = root / "handeye_eval_report.txt";
        std::ofstream reportFile(reportPath);
        if (!reportFile) {
            throw std::runtime_error("failed to create report file: " + reportPath.string());
        }
        evaluator.writeReport(report, reportFile);
        evaluator.writeReport(report, std::cout);
        std::cout << "report_path=" << reportPath.string() << '\n';
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}

