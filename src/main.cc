/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "benchmark/backendbench.h"
#include "benchmark/benchmark.h"
#include "chess/board.h"
#include "engine.h"
#include "lc0ctl/describenet.h"
#include "lc0ctl/leela2onnx.h"
#include "lc0ctl/onnx2leela.h"
#include "selfplay/loop.h"
#include "utils/commandline.h"
#include "utils/esc_codes.h"
#include "utils/logging.h"
#include "version.h"
#include <vector>

struct BotDifficultySettings {
  std::int64_t movetime;
  int depth;
};

BotDifficultySettings getMovetime(float difficulty) {
  if (difficulty <= 0) {
    return {20, 1};
  }
  if (difficulty <= 0.2) {
    return {20, 1};
  }
  if (difficulty <= 0.3) {
    return {50, 2};
  }
  if (difficulty <= 0.4) {
    return {100, 3};
  }
  if (difficulty <= 0.5) {
    return {150, 4};
  }
  if (difficulty <= 0.6) {
    return {200, 5};
  }
  if (difficulty <= 0.7) {
    return {300, 5};
  }
  if (difficulty <= 0.8) {
    return {400, 8};
  }
  if (difficulty <= 0.9) {
    return {500, 13};
  }
  return {600, 18};
}

int main(int argc, const char** argv) {
  using namespace lczero;
  EscCodes::Init();
  LOGFILE << "Lc0 started.";
  CERR << EscCodes::Bold() << EscCodes::Red() << "       _";
  CERR << "|   _ | |";
  CERR << "|_ |_ |_|" << EscCodes::Reset() << " v" << GetVersionStr()
       << " built " << __DATE__;

  try {
    InitializeMagicBitboards();

    OptionsParser parser;
    EngineController engine(
        std::make_unique<CallbackUciResponder>(
            [](const BestMoveInfo& info) {
              std::cout << "BEST MOVE: " << info.bestmove.from().as_string() << " : " << info.bestmove.to().as_string() << std::endl;
            },
            [](const std::vector<ThinkingInfo>& info) {
              std::cout << "THINKING..." << std::endl;
              for (const auto &row : info) {
                std::cout << "row: " << row.depth << ", " << row.score.value() << ", "
                          << row.comment << std::endl;
              }
            }),
        parser.GetOptionsDict());
    engine.PopulateOptions(&parser);

    // Ordinary UCI engine.
    // EngineLoop loop;
//      loop.RunLoop();
    std::cout << "Starting new game" << std::endl;
    // loop.CmdUciNewGame();
    engine.NewGame();
    std::cout << "Starting position" << std::endl;
    // g1 -> f3 seems to be best move from this position
    engine.SetPosition("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", {});
    const auto difficulty = getMovetime(1.0);
    GoParams goParams;
    goParams.movetime = difficulty.movetime;
    goParams.depth = difficulty.depth;
    engine.Go(goParams);

    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2'000));
      std::cout << "Waiting..." << std::endl;
    }

  } catch (std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    abort();
  }

  return 0;
}
