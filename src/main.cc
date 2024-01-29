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

// We need models for the following ranges:
// 250, 500, 750, 1000, 1400, 1700, 1900, 2100, 2300, 2700
const std::string EngineWeight_Maia_1100 = "maia-1100.pb"; // Plays at around 1550
const std::string EngineWeight_Maia_1500 = "maia-1500.pb"; // Plays at around 1600ish?
const std::string EngineWeight_Maia_1900 = "maia-1900.pb"; // Plays at around 1700
const std::string EngineWeight_Elo_206 = "elo-206";
const std::string EngineWeight_Elo_416 = "elo-416";
const std::string EngineWeight_Elo_754 = "elo-754";
const std::string EngineWeight_Elo_999 = "elo-999";
const std::string EngineWeight_Elo_2100 = "elo-2100";
const std::string EngineWeight_Elo_2304 = "elo-2304";
const std::string EngineWeight_Elo_2701 = "elo-2701";

struct EngineWithOptions {
  lczero::OptionsParser parser;
  lczero::EngineController controller;

  EngineWithOptions(std::unique_ptr<lczero::UciResponder> uciResponder)
      : controller(std::move(uciResponder), parser.GetOptionsDict()) {
    controller.PopulateOptions(&parser);
  }
};

/** Maps weights -> the engine for that weight. */
std::unordered_map<std::string, EngineWithOptions> weightToEngine;

struct BotDifficultySettings {
  /** Time in milliseconds allowed for making a move. Lower values will make the model play worse. */
  std::int64_t movetime;
  int depth;
  int nodes;
  /** Tau value from softmax between [0, 1]. Higher value makes it more random. Value of 0 means "always best move". */
  float temperature;
  std::string model;
};

std::string getPromotionAsString(lczero::Move::Promotion promotion) {
  using namespace lczero;
  switch (promotion) {
    case Move::Promotion::Bishop:
      return "b";
    case Move::Promotion::Knight:
      return "n";
    case Move::Promotion::Queen:
      return "q";
    case Move::Promotion::Rook:
      return "r";
  }
  return "";
}

std::string infoToJsonString(const lczero::BestMoveInfo& info) {
  std::stringstream ss;
  ss << "{";
  ss << "\"result\":{";

  ss << "\"from\":\"" << info.bestmove.from().as_string() << "\"";
  ss << ",\"to\":\"" << info.bestmove.to().as_string() << "\"";
  const std::string& promotion =
      getPromotionAsString(info.bestmove.promotion());
  if (!promotion.empty()) {
    ss << ",\"promotion\":\"" << getPromotionAsString(info.bestmove.promotion())
       << "\"";
  }

  ss << "}";
  ss << "}";
  return ss.str();
}

lczero::EngineController* getOrCreateEngine(
    const BotDifficultySettings& settings, std::optional<httplib::Response*>& optionalRes) {
  using namespace lczero;
  const auto& weight = settings.model;
  if (weightToEngine.find(weight) == weightToEngine.end()) {
    weightToEngine.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(weight),
        std::forward_as_tuple(
            std::make_unique<CallbackUciResponder>(
                [&optionalRes](const BestMoveInfo& info) {
                  if (optionalRes.has_value()) {
                    optionalRes.value()->set_content(infoToJsonString(info), "application/json");
                    optionalRes.reset();
                  } else {
                    std::cerr << "EXPECTED OPTIONAL RES TO EXIST, BUT IT DIDN'T! RACE CONDITION???\n";
                    abort();
                  }
                },
                [](const std::vector<ThinkingInfo>& infos) {
                  for (const auto &info : infos) {
                    std::cout << "Score: " << info.score.value() << "\n";
                    for (const auto &p : info.pv) {
                      std::cout << "move: " << p.as_string() << "\n";
                    }
                  }
                })
        ));
    auto& parser = weightToEngine.at(weight).parser;
    parser.SetUciOption("WeightsFile", weight);
    // parser.SetUciOption("MultiPV", "5");  // Displays the top 5 moves. Right now score seems broken and only reports the top score?
    // parser.SetUciOption("ScoreType", "centipawn");
    parser.SetUciOption("Temperature", std::to_string(settings.temperature));
    // Default cache size is 200,000 which results in ~800 MB total memory usage w/ 10 models. By using 20,000 we're only at 200 MB
    parser.SetUciOption("NNCacheSize", "20000");
  }

  return &weightToEngine.at(weight).controller;
}

BotDifficultySettings getMovetime(float difficulty) {
  // Note: We limit the nodes (third parameter), which makes these lookups super fast
  if (difficulty <= 0.1) {
    return {200, 1, 1, 0.5f, EngineWeight_Elo_206};
  }
  if (difficulty <= 0.2) {
    // This model by itself is way stronger than 700, so we nerf it with higher temperature
    return {200, 1, 1, 0.5f, EngineWeight_Elo_416};
  }
  if (difficulty <= 0.3) {
    return {200, 2, 1, 0.5f, EngineWeight_Elo_754};
  }
  if (difficulty <= 0.4) {
    return {200, 30, 0, 0.5f, EngineWeight_Elo_999};
  }
  if (difficulty <= 0.5) {
    return {200, 4, 1, 0.5f, EngineWeight_Maia_1100};
  }
  if (difficulty <= 0.6) {
    return {200, 0, 1, 0.5f, EngineWeight_Maia_1500};
  }
  if (difficulty <= 0.7) {
    return {200, 0, 1, 0.5f, EngineWeight_Maia_1900};  // Seems about right. Lost to Chef Magnus (2000)
  }
  if (difficulty <= 0.8) {
    return {200, 8, 1, 0.5f, EngineWeight_Elo_2100};  // Seems about right. Won against Chef Magnus (2000)
  }
  if (difficulty <= 0.9) {
    return {200, 13, 1, 0.5f, EngineWeight_Elo_2304};
  }
  return {200, 18, 1, 0.5f, EngineWeight_Elo_2701};  // Seems good. Lost to Magnus bot (2880)
}

class Date {
 public:
  /** Returns the current Unix timestamp since the Unix Epoch in milliseconds.
   */
  static long long now() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  }
};

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

  Mutex engineMutex;
  httplib::Server server;

  // Start up all bot difficulties
  for (const float difficulty : {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f}) {
    getOrCreateEngine(getMovetime(difficulty), optionalRes)->NewGame();
  }
  std::cout << "Done initializing\n";

  server.Get("/", [&engineMutex, &optionalRes](const httplib::Request& req,
                                               httplib::Response& res) {
    Mutex::Lock lock(engineMutex);  // Ensure we're only running one instance of
                                    // engine at a time

    const std::string fen = req.get_param_value("fen");
    std::cout << "Request fen: " << req.get_param_value("fen") << "\n";
    std::cout << "Request difficulty: " << req.get_param_value("difficulty") << "\n";

    try {
      const BotDifficultySettings settings =
          getMovetime(std::stof(req.get_param_value("difficulty")));
      const auto engine = getOrCreateEngine(settings, optionalRes);
      engine->NewGame();
      engine->SetPosition(fen, {});
      GoParams goParams;
      goParams.movetime = settings.movetime;
      if (settings.depth > 0) {
        goParams.depth = settings.depth;
      }
      if (settings.nodes > 0) {
        goParams.nodes = settings.nodes;
      }
      optionalRes = &res;
      engine->Go(goParams);

      const auto start = Date::now();
      while (optionalRes.has_value() && optionalRes.value() == &res) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::cout << "Duration: " << (Date::now() - start) << "ms\n\n";
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
