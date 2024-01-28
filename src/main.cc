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
#include "net/httplib.h"
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

class Date {
 public:
  /** Returns the current Unix timestamp since the Unix Epoch in milliseconds. */
  static long long now() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
};

std::string getPromotionAsString(lczero::Move::Promotion promotion) {
  using namespace lczero;
  switch (promotion) {
    case Move::Promotion::Bishop: return "b";
    case Move::Promotion::Knight: return "n";
    case Move::Promotion::Queen: return "q";
    case Move::Promotion::Rook: return "r";
  }
  return "";
}

int main(int argc, const char** argv) {
  using namespace lczero;
  EscCodes::Init();
  LOGFILE << "Lc0 started.";
  CERR << EscCodes::Bold() << EscCodes::Red() << "       _";
  CERR << "|   _ | |";
  CERR << "|_ |_ |_|" << EscCodes::Reset() << " v" << GetVersionStr()
       << " built " << __DATE__;

  InitializeMagicBitboards();

  std::optional<httplib::Response *> optionalRes;
  OptionsParser parser;
  long long start;
  EngineController engine(
      std::make_unique<CallbackUciResponder>(
          [&optionalRes, &start](const BestMoveInfo& info) {
            if (optionalRes.has_value()) {
              std::stringstream ss;
              ss << "{";
              ss << "\"result\":{";

              ss << "\"from\":\"" << info.bestmove.from().as_string() << "\"";
              ss << ",\"to\":\"" << info.bestmove.to().as_string() << "\"";
              const std::string& promotion = getPromotionAsString(info.bestmove.promotion());
              if (!promotion.empty()) {
                ss << ",\"promotion\":\"" << getPromotionAsString(info.bestmove.promotion()) << "\"";
              }

              ss << "}";
              ss << "}";
              optionalRes.value()->set_content(ss.str(), "application/json");
              optionalRes.reset();
              std::cout << "Done: " << (Date::now() - start) << "ms\n";
            } else {
              std::cerr << "EXPECTED OPTIONAL RES TO EXIST, BUT IT DIDN'T! RACE CONDITION???\n";
              abort();
            }
          },
          [](const std::vector<ThinkingInfo>& info) { /* We don't care about when it's thinking */ }),
      parser.GetOptionsDict());
  engine.PopulateOptions(&parser);

  Mutex engineMutex;
  httplib::Server server;
  server.Get("/", [&start, &engine, &engineMutex, &optionalRes](const httplib::Request &req, httplib::Response &res) {
    Mutex::Lock lock(engineMutex);  // Ensure we're only running one instance of engine at a time

    try {
      std::cout << "Starting new game: " << Date::now() << "\n";
      start = Date::now();
      engine.NewGame();
      // g1 -> f3 seems to be best move from this position
      engine.SetPosition(
          "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", {});
      const auto difficulty = getMovetime(0.4);
      GoParams goParams;
      goParams.movetime = difficulty.movetime;
      goParams.depth = difficulty.depth;
      optionalRes = &res;
      engine.Go(goParams);

      while (optionalRes.has_value() && optionalRes.value() == &res) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } catch (std::exception& e) {
      std::cerr << "Unhandled exception: " << e.what() << "\n";
      abort();
    }
  });
  server.listen("0.0.0.0", 3002);

  std::cout << "Listening on port 3000...\n";
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2'000));
  }
  
  return 0;
}
