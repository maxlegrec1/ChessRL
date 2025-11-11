#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "inference.hpp"

int main()
{
    try
    {
        const std::string fen = "r3k2r/ppp2ppp/2P5/8/2p1q1b1/B1P2N2/P4PPP/R2Q2K1 b kq - 0 14";
        constexpr int kBatchSize = 1024;
        std::vector<std::string> batch(kBatchSize, fen);
        const float temperature = 0.1F;
        constexpr int kIterations = 10;

        chessrl::MoveGenerationResult result = chessrl::generate_moves(batch, temperature, kIterations);

        std::cout << "Generated " << result.moves.size() << " moves (from final iteration):" << std::endl;
        for (size_t i = 0; i < result.moves.size(); ++i)
        {
            std::cout << result.moves[i];
            if ((i + 1) % 16 == 0)
            {
                std::cout << '\n';
            }
            else
            {
                std::cout << ' ';
            }
        }
        if (result.moves.size() % 16 != 0)
        {
            std::cout << std::endl;
        }

        std::cout << std::fixed;
        std::cout << "Average timings over " << kIterations << " runs (ms):" << std::endl;
        std::cout << "  encode: " << result.timings.encodeMs << std::endl;
        std::cout << "  inference: " << result.timings.inferMs << std::endl;
        std::cout << "  sampling: " << result.timings.sampleMs << std::endl;
        std::cout << "  total: " << result.timings.totalMs << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
