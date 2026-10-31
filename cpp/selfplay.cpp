#include "selfplay.hpp"
#include "inference.hpp"
#include "mcts/common.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/cache.hpp"
#include "chess-library/include/chess.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace chessrl
{

namespace
{

// Searches each self-play thread interleaves (CHESSRL_SEARCHES_PER_THREAD, default 1,
// run_small_mcts_batch_shared); the runner's batch threshold counts them all.
int searches_per_thread()
{
    static const int n = [] {
        const char* env = std::getenv("CHESSRL_SEARCHES_PER_THREAD");
        return std::max(1, env ? std::atoi(env) : 1);
    }();
    return n;
}

// On-disk record layout v2 (1280 bytes, fixed). Bytes [0..832) keep the v1
// layout; v2 appends the current position's bitboards and scalar state (so the
// trainer builds input planes without parsing FEN) and the MCTS visit
// distribution (the soft policy target).
//   [0..100)    FEN (null-padded)
//   [100..116)  played move, UCI (null-padded)
//   [116..128)  value_outcome (3 x f32), side-to-move WDL from the game result
//   [128..140)  value_mcts    (3 x f32), root WDL after search
//   [140..144)  plies_remaining (f32), set at game end
//   [144..148)  backed_plies (f32), MCTS-backed plies estimate at root
//   [148..155)  past repetition flags (7 x u8), slots t-1..t-7
//   [155]       current repetition flag (u8)
//   [156..160)  padding
//   [160..832)  past bitboards: 7 slots x 12 u64 (W:PNBRQK, B:PNBRQK), newest first
//   [832..928)  current bitboards: 12 u64, same order
//   [928]       record version (u8) = 2
//   [929]       flags (u8), kFlag* below
//   [930]       castling rights (u8): WK=1, WQ=2, BK=4, BQ=8
//   [931]       en-passant square 0..63 (a1=0), 255 = none
//   [932..934)  halfmove clock (u16)
//   [934..936)  fullmove number (u16)
//   [936..938)  ply index within the game, MCTS plies (u16)
//   [938..940)  simulations used for this search (u16)
//   [940..942)  number of policy entries K (u16)
//   [942..944)  played move vocab index (u16), 0xFFFF = not in vocab
//   [944..1280) K <= 84 policy entries (u16 vocab index, u16 visit fraction * 65535),
//               sorted by visits, descending. 84 covers virtually every chess
//               position; a top-20 cap kept only ~74% of the visit mass early on,
//               when an untrained net spreads visits over every legal move.
constexpr size_t kRecordSize = 1280;
constexpr size_t kFenOffset = 0;
constexpr size_t kPolicyOffset = 100;
constexpr size_t kValueOutcomeOffset = 116;
constexpr size_t kValueMctsOffset = 128;
constexpr size_t kPliesRemainingOffset = 140;
constexpr size_t kBackedPliesOffset = 144;
constexpr size_t kPastRepetitionOffset = 148;
constexpr size_t kCurrentRepetitionOffset = 155;
constexpr size_t kPastBitboardsOffset = 160;
constexpr size_t kBitboardsPerSlot = 12;
constexpr size_t kHistoricalSlots = 7;
constexpr size_t kBitboardSlotBytes = kBitboardsPerSlot * sizeof(std::uint64_t); // 96
constexpr size_t kCurrentBitboardsOffset = 832;
constexpr size_t kVersionOffset = 928;
constexpr size_t kFlagsOffset = 929;
constexpr size_t kCastlingOffset = 930;
constexpr size_t kEpOffset = 931;
constexpr size_t kHalfmoveOffset = 932;
constexpr size_t kFullmoveOffset = 934;
constexpr size_t kPlyOffset = 936;
constexpr size_t kSimsOffset = 938;
constexpr size_t kPolicyCountOffset = 940;
constexpr size_t kPlayedIndexOffset = 942;
constexpr size_t kPolicyEntriesOffset = 944;
constexpr size_t kMaxPolicyEntries = 84;
static_assert(kPastBitboardsOffset + kHistoricalSlots * kBitboardSlotBytes == kCurrentBitboardsOffset,
              "record layout math is inconsistent");
static_assert(kCurrentBitboardsOffset + kBitboardSlotBytes == kVersionOffset,
              "record layout math is inconsistent");
static_assert(kPolicyEntriesOffset + kMaxPolicyEntries * 4 == kRecordSize,
              "record layout math is inconsistent");

constexpr std::uint8_t kRecordVersion = 2;
constexpr std::uint8_t kFlagFullSearch = 1;   // policy target comes from a full search
constexpr std::uint8_t kFlagSampledMove = 2;  // move chosen with root temperature > 0
constexpr std::uint8_t kFlagResigned = 4;     // game ended by resignation
constexpr std::uint8_t kFlagPlaythrough = 8;  // resignation disabled in this game
constexpr std::uint8_t kFlagCurriculum = 16;  // game started from a curriculum position
constexpr std::uint8_t kFlagMaxPlies = 32;    // game adjudicated a draw at max_plies

enum class EndReason
{
    Checkmate,
    Stalemate,
    Insufficient,
    Fifty,
    Repetition,
    MaxPlies,
    Resign,
};

const std::unordered_map<std::string, int>& vocab_index()
{
    static const std::unordered_map<std::string, int> index = [] {
        std::unordered_map<std::string, int> m;
        const auto& vocab = small_policy_vocabulary();
        m.reserve(vocab.size());
        for (size_t i = 0; i < vocab.size(); ++i)
        {
            m.emplace(vocab[i], static_cast<int>(i));
        }
        return m;
    }();
    return index;
}

// Same key convention as the MCTS prior lookup: knight promotions share the
// suffix-less vocab entry.
int vocab_index_of(std::string uci)
{
    if (!uci.empty() && uci.back() == 'n')
    {
        uci.pop_back();
    }
    const auto& m = vocab_index();
    auto it = m.find(uci);
    return it == m.end() ? -1 : it->second;
}

chess::Color opposite(chess::Color c)
{
    return c == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE;
}

} // namespace

struct SelfPlayRecord
{
    std::string fen;
    std::string policy;
    // Both WDLs are stored from the side-to-move's perspective.
    std::array<float, 3> value_outcome{}; // sparse one-hot, set at game end from z
    std::array<float, 3> value_mcts{};    // dense WDL from MCTS root after refine
    int ply_index{0};                     // 0-based ply index within the game
    float plies_remaining{-1.0f};         // total_plies - ply_index, set at game end
    float backed_plies{-1.0f};            // MCTS-backed plies-remaining at root
    // AlphaZero-style history: 7 past piece-plane snapshots (index 0 = t-1
    // .. index 6 = t-7) plus a per-slot repetition flag and a flag for the
    // current position. All zeros means "no history available".
    std::array<std::array<std::uint64_t, 12>, 7> past_bitboards{};
    std::array<std::uint8_t, 7> past_repetition_flags{};
    std::uint8_t current_repetition_flag{0};
    // v2 fields.
    bool white_to_move{true};
    std::array<std::uint64_t, 12> current_bitboards{};
    std::uint8_t flags{0};
    std::uint8_t castling{0};
    std::uint8_t ep_square{255};
    std::uint16_t halfmove{0};
    std::uint16_t fullmove{1};
    std::uint16_t simulations{0};
    std::uint16_t played_index{0xFFFF};
    std::vector<std::pair<std::uint16_t, std::uint16_t>> policy_entries;
};

struct BoardSlot
{
    chess::Board board;
    std::vector<std::string> history;
    std::vector<SelfPlayRecord> pending_records;
    int ply_count = 0;
    // Rolling deque of recent past piece-plane snapshots (newest first).
    // At record-creation time, the first 7 entries describe positions t-1
    // .. t-7 relative to the current `board`. Pushed on every makeMove,
    // trimmed to at most 7 entries.
    std::deque<chessrl::mcts::PiecePlanes> past_planes;
    bool playthrough = false;                 // resignation disabled for this game
    bool curriculum = false;                  // started from a curriculum position
    std::array<int, 2> low_streak{{0, 0}};    // consecutive below-threshold moves (0 = white)
    std::array<bool, 2> would_resign{{false, false}};
    // tree_reuse: the subtree of the move just played, for this game's next search.
    std::shared_ptr<chessrl::mcts::SearchTree> tree;
};

struct SelfPlayWorker::Impl
{
    SelfPlayConfig config;
    chessrl::mcts::single::SmallBatchConfig batch_config;
    std::shared_ptr<chessrl::mcts::single::SmallBatchRunner> runner;
    std::shared_ptr<chessrl::mcts::TranspositionTable> cache;
    std::vector<BoardSlot> slots;
    std::fstream outfile;
    size_t current_records = 0;
    // Monotonic counter of EVERY write (append + replacement). Once the
    // buffer is at max_buffer_size the file size stops growing, so the
    // trainer can no longer infer insertion rate from the file. We persist
    // this counter to a sidecar `<output_path>.writes` after each flush so
    // the trainer can read insertion rate as delta(total_writes) / dt. With
    // fifo_buffer the newest record sits at (total_writes - 1) % capacity.
    size_t total_writes = 0;
    std::mt19937 opening_rng{std::random_device{}()};
    std::uniform_real_distribution<float> unit{0.0f, 1.0f};
    SelfPlayStats stats;

    float uniform01() { return unit(opening_rng); }

    // A random legal position with both kings and 1..curriculum_max_pieces other
    // pieces, side to move random, no castling or en passant. Rules only: no
    // human games or engine knowledge involved.
    bool make_curriculum_board(chess::Board& out)
    {
        static constexpr char kTypes[] = {'q', 'r', 'r', 'b', 'n', 'p', 'p'};
        std::uniform_int_distribution<int> sq_dist(0, 63);
        std::uniform_int_distribution<int> type_dist(0, static_cast<int>(sizeof(kTypes)) - 1);
        std::uniform_int_distribution<int> count_dist(1, std::max(1, config.curriculum_max_pieces));
        std::bernoulli_distribution coin(0.5);
        auto chebyshev = [](int a, int b) { return std::max(std::abs(a % 8 - b % 8), std::abs(a / 8 - b / 8)); };

        for (int attempt = 0; attempt < 200; ++attempt)
        {
            std::array<char, 64> grid;
            grid.fill('.');
            const int wk = sq_dist(opening_rng);
            const int bk = sq_dist(opening_rng);
            if (chebyshev(wk, bk) <= 1) continue;
            grid[static_cast<size_t>(wk)] = 'K';
            grid[static_cast<size_t>(bk)] = 'k';

            const int n = count_dist(opening_rng);
            for (int p = 0; p < n; ++p)
            {
                const char type = kTypes[type_dist(opening_rng)];
                const bool white = coin(opening_rng);
                int sq = -1;
                for (int tries = 0; tries < 64 && sq < 0; ++tries)
                {
                    const int cand = sq_dist(opening_rng);
                    if (grid[static_cast<size_t>(cand)] != '.') continue;
                    if (type == 'p' && (cand / 8 == 0 || cand / 8 == 7)) continue;
                    sq = cand;
                }
                if (sq < 0) continue;
                grid[static_cast<size_t>(sq)] = white ? static_cast<char>(std::toupper(type)) : type;
            }

            // Square index = rank * 8 + file (a1 = 0), FEN lists rank 8 first.
            std::string fen;
            for (int rank = 7; rank >= 0; --rank)
            {
                int empty = 0;
                for (int file = 0; file < 8; ++file)
                {
                    const char c = grid[static_cast<size_t>(rank * 8 + file)];
                    if (c == '.')
                    {
                        ++empty;
                        continue;
                    }
                    if (empty) fen += static_cast<char>('0' + empty);
                    empty = 0;
                    fen += c;
                }
                if (empty) fen += static_cast<char>('0' + empty);
                if (rank) fen += '/';
            }
            fen += coin(opening_rng) ? " w - - 0 1" : " b - - 0 1";

            chess::Board b(fen);
            const chess::Color stm = b.sideToMove();
            if (b.isAttacked(b.kingSq(opposite(stm)), stm)) continue; // side not to move in check
            if (b.isGameOver().second != chess::GameResult::NONE) continue;
            out = b;
            return true;
        }
        return false;
    }

    // Reset a slot to a fresh starting position: a curriculum position with
    // probability curriculum_prob, otherwise the start position optionally
    // seeded with N random legal plies for opening diversity. The random plies
    // are added to slot.history (so MCTS gets the correct move context) but no
    // SelfPlayRecord entries are produced for them.
    void reset_slot(BoardSlot& slot)
    {
        slot.board = chess::Board();
        slot.tree.reset();
        slot.history.clear();
        slot.pending_records.clear();
        slot.ply_count = 0;
        slot.past_planes.clear();
        slot.low_streak = {{0, 0}};
        slot.would_resign = {{false, false}};
        slot.playthrough = config.resign_threshold > 0.0f && uniform01() < config.resign_playthrough;
        slot.curriculum = false;

        if (config.curriculum_prob > 0.0f && uniform01() < config.curriculum_prob)
        {
            chess::Board b;
            if (make_curriculum_board(b))
            {
                slot.board = b;
                slot.curriculum = true;
                return;
            }
        }

        const int n = config.random_opening_plies;
        for (int i = 0; i < n; ++i)
        {
            if (slot.board.isGameOver().second != chess::GameResult::NONE) break;
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, slot.board);
            if (moves.size() == 0) break;
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            const auto& m = moves[dist(opening_rng)];
            // Snapshot the pre-move position into the past_planes deque so
            // that subsequent records carry the random opening as history.
            chessrl::mcts::PiecePlanes pre = chessrl::mcts::snapshot_piece_planes(slot.board);
            pre.repetition = slot.board.isRepetition(1);
            slot.past_planes.push_front(pre);
            if (slot.past_planes.size() > 7) slot.past_planes.pop_back();

            slot.history.push_back(chess::uci::moveToUci(m, slot.board.chess960()));
            slot.board.makeMove(m);
        }
    }

    SelfPlayRecord make_record(const BoardSlot& slot, const chessrl::mcts::MctsSummary& summary,
                               int simulations, bool full_search, bool sampled_move) const
    {
        SelfPlayRecord record;
        record.fen = slot.board.getFen();
        record.policy = summary.action;
        // value_mcts is the MCTS-backed-up WDL (always populated).
        // value_outcome is filled in at game end; zero until then.
        record.value_mcts = summary.value;
        record.ply_index = slot.ply_count;
        // backed_plies is the search's plies-remaining estimate at this
        // position; negative sentinel if MCTS produced none.
        record.backed_plies = summary.backed_plies;

        for (size_t k = 0; k < slot.past_planes.size() && k < 7; ++k)
        {
            const auto& past = slot.past_planes[k];
            record.past_bitboards[k] = past.bitboards;
            record.past_repetition_flags[k] = past.repetition ? 1u : 0u;
        }
        record.current_repetition_flag = slot.board.isRepetition(1) ? 1u : 0u;

        record.white_to_move = slot.board.sideToMove() == chess::Color::WHITE;
        record.current_bitboards = chessrl::mcts::snapshot_piece_planes(slot.board).bitboards;
        record.castling = static_cast<std::uint8_t>(slot.board.castlingRights().hashIndex());
        const chess::Square ep = slot.board.enpassantSq();
        record.ep_square = ep == chess::Square::NO_SQ ? 255 : static_cast<std::uint8_t>(ep.index());
        record.halfmove = static_cast<std::uint16_t>(std::min<std::uint32_t>(slot.board.halfMoveClock(), 0xFFFF));
        record.fullmove = static_cast<std::uint16_t>(std::min<std::uint32_t>(slot.board.fullMoveNumber(), 0xFFFF));
        record.simulations = static_cast<std::uint16_t>(std::min(simulations, 0xFFFF));
        record.flags = static_cast<std::uint8_t>((full_search ? kFlagFullSearch : 0) |
                                                 (sampled_move ? kFlagSampledMove : 0) |
                                                 (slot.playthrough ? kFlagPlaythrough : 0) |
                                                 (slot.curriculum ? kFlagCurriculum : 0));
        const int played = vocab_index_of(summary.action);
        record.played_index = played < 0 ? 0xFFFF : static_cast<std::uint16_t>(played);

        // summary.policy is sorted by visit fraction, descending.
        record.policy_entries.reserve(kMaxPolicyEntries);
        for (const auto& [uci, fraction] : summary.policy)
        {
            if (record.policy_entries.size() >= kMaxPolicyEntries) break;
            const int idx = vocab_index_of(uci);
            const long q = std::lround(static_cast<double>(fraction) * 65535.0);
            if (idx < 0 || q <= 0) continue;
            record.policy_entries.emplace_back(static_cast<std::uint16_t>(idx),
                                               static_cast<std::uint16_t>(std::min(q, 65535L)));
        }
        return record;
    }

    void write_record(const SelfPlayRecord& rec)
    {
        char buffer[kRecordSize];
        std::memset(buffer, 0, kRecordSize);

        std::strncpy(buffer + kFenOffset, rec.fen.c_str(), 99);
        std::strncpy(buffer + kPolicyOffset, rec.policy.c_str(), 15);
        std::memcpy(buffer + kValueOutcomeOffset, rec.value_outcome.data(), 12);
        std::memcpy(buffer + kValueMctsOffset, rec.value_mcts.data(), 12);
        std::memcpy(buffer + kPliesRemainingOffset, &rec.plies_remaining, 4);
        std::memcpy(buffer + kBackedPliesOffset, &rec.backed_plies, 4);
        std::memcpy(buffer + kPastRepetitionOffset, rec.past_repetition_flags.data(),
                    rec.past_repetition_flags.size());
        buffer[kCurrentRepetitionOffset] = static_cast<char>(rec.current_repetition_flag);
        for (size_t k = 0; k < kHistoricalSlots; ++k)
        {
            std::memcpy(buffer + kPastBitboardsOffset + k * kBitboardSlotBytes,
                        rec.past_bitboards[k].data(), kBitboardSlotBytes);
        }
        std::memcpy(buffer + kCurrentBitboardsOffset, rec.current_bitboards.data(), kBitboardSlotBytes);
        buffer[kVersionOffset] = static_cast<char>(kRecordVersion);
        buffer[kFlagsOffset] = static_cast<char>(rec.flags);
        buffer[kCastlingOffset] = static_cast<char>(rec.castling);
        buffer[kEpOffset] = static_cast<char>(rec.ep_square);
        const std::uint16_t ply = static_cast<std::uint16_t>(std::clamp(rec.ply_index, 0, 0xFFFF));
        const std::uint16_t n_policy = static_cast<std::uint16_t>(rec.policy_entries.size());
        std::memcpy(buffer + kHalfmoveOffset, &rec.halfmove, 2);
        std::memcpy(buffer + kFullmoveOffset, &rec.fullmove, 2);
        std::memcpy(buffer + kPlyOffset, &ply, 2);
        std::memcpy(buffer + kSimsOffset, &rec.simulations, 2);
        std::memcpy(buffer + kPolicyCountOffset, &n_policy, 2);
        std::memcpy(buffer + kPlayedIndexOffset, &rec.played_index, 2);
        for (size_t k = 0; k < rec.policy_entries.size(); ++k)
        {
            std::memcpy(buffer + kPolicyEntriesOffset + 4 * k, &rec.policy_entries[k].first, 2);
            std::memcpy(buffer + kPolicyEntriesOffset + 4 * k + 2, &rec.policy_entries[k].second, 2);
        }

        size_t idx;
        if (current_records < config.max_buffer_size)
        {
            idx = current_records++;
        }
        else if (config.fifo_buffer)
        {
            idx = total_writes % config.max_buffer_size;
        }
        else
        {
            std::uniform_int_distribution<size_t> dist(0, config.max_buffer_size - 1);
            idx = dist(opening_rng);
        }
        outfile.seekp(static_cast<std::streamoff>(idx * kRecordSize), std::ios::beg);
        outfile.write(buffer, kRecordSize);
        // Count every write, whichever branch placed it: the trainer derives
        // the insertion rate (and the FIFO head) from this counter.
        total_writes++;
        stats.records_written++;
    }

    void write_sidecars()
    {
        // Atomic rename so a partial write is never observed.
        auto atomic_write = [](const std::string& path, const std::string& text) {
            const std::string tmp = path + ".tmp";
            {
                std::ofstream out(tmp, std::ios::trunc);
                if (!out) return;
                out << text;
            }
            std::error_code ec;
            std::filesystem::rename(tmp, path, ec);
        };
        atomic_write(config.output_path + ".writes", std::to_string(total_writes) + "\n");

        std::ostringstream js;
        js << "{\"record_size\": " << kRecordSize
           << ", \"capacity\": " << config.max_buffer_size
           << ", \"fifo\": " << (config.fifo_buffer ? "true" : "false")
           << ", \"total_writes\": " << total_writes
           << ", \"games\": " << stats.games
           << ", \"white_wins\": " << stats.white_wins
           << ", \"black_wins\": " << stats.black_wins
           << ", \"draws\": " << stats.draws
           << ", \"game_plies\": " << stats.game_plies
           << ", \"end_checkmate\": " << stats.end_checkmate
           << ", \"end_stalemate\": " << stats.end_stalemate
           << ", \"end_insufficient\": " << stats.end_insufficient
           << ", \"end_fifty\": " << stats.end_fifty
           << ", \"end_repetition\": " << stats.end_repetition
           << ", \"end_max_plies\": " << stats.end_max_plies
           << ", \"end_resign\": " << stats.end_resign
           << ", \"playthrough_games\": " << stats.playthrough_games
           << ", \"playthrough_would_resign\": " << stats.playthrough_would_resign
           << ", \"playthrough_false_resign\": " << stats.playthrough_false_resign
           << ", \"curriculum_games\": " << stats.curriculum_games
           << ", \"full_searches\": " << stats.full_searches
           << ", \"fast_searches\": " << stats.fast_searches
           << ", \"simulations_run\": " << stats.simulations_run
           << ", \"reused_visits\": " << stats.reused_visits
           << ", \"policy_moves\": " << stats.policy_moves
           << ", \"records_written\": " << stats.records_written
           << ", \"engine_swaps\": " << stats.engine_swaps
           << ", \"unix_time\": "
           << std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count()
           << "}\n";
        atomic_write(config.output_path + ".stats.json", js.str());
    }

    void write_pgn(const BoardSlot& slot, float z, size_t slot_index)
    {
        std::filesystem::create_directories(config.pgn_dir);
        std::string result_str = "1/2-1/2";
        if (z == 1.0f) result_str = "1-0";
        else if (z == 0.0f) result_str = "0-1";

        std::stringstream ss;
        ss << "[Event \"Self-Play\"]\n[Site \"Local\"]\n[Date \"????.??.??\"]\n[Round \"-\"]\n"
           << "[White \"SelfPlay\"]\n[Black \"SelfPlay\"]\n[Result \"" << result_str << "\"]\n\n";

        // Replay from the standard start to produce SAN. Curriculum games start
        // elsewhere; their history only covers moves since that position.
        chess::Board replay_board;
        int move_num = 1;
        for (size_t k = 0; k < slot.history.size(); ++k)
        {
            if (k % 2 == 0) ss << move_num << ". ";
            bool found = false;
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, replay_board);
            for (const auto& leg_m : moves)
            {
                if (chess::uci::moveToUci(leg_m, replay_board.chess960()) == slot.history[k])
                {
                    ss << chess::uci::moveToSan(replay_board, leg_m) << " ";
                    replay_board.makeMove(leg_m);
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ss << slot.history[k] << "? ";
                break;
            }
            if (k % 2 == 1) move_num++;
        }
        ss << result_str << "\n";

        const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::ofstream pgn_file(config.pgn_dir + "/game_" + std::to_string(timestamp) + "_" +
                               std::to_string(slot_index) + ".pgn");
        if (pgn_file) pgn_file << ss.str();
    }

    // z = 1 white wins, 0 black wins, 0.5 draw.
    void finish_game(BoardSlot& slot, size_t slot_index, float z, EndReason reason)
    {
        const int total_plies = slot.ply_count;
        std::uint8_t game_flags = 0;
        if (reason == EndReason::Resign) game_flags |= kFlagResigned;
        if (reason == EndReason::MaxPlies) game_flags |= kFlagMaxPlies;

        // Sparse one-hot game-outcome WDL from each record's side to move, plus
        // the plies-remaining regression target for the moves-left head.
        for (auto& rec : slot.pending_records)
        {
            const float stm_score = rec.white_to_move ? z : 1.0f - z;
            if (stm_score == 1.0f) rec.value_outcome = {1.0f, 0.0f, 0.0f};
            else if (stm_score == 0.0f) rec.value_outcome = {0.0f, 0.0f, 1.0f};
            else rec.value_outcome = {0.0f, 1.0f, 0.0f};
            rec.plies_remaining = static_cast<float>(std::max(0, total_plies - rec.ply_index));
            rec.flags |= game_flags;
            write_record(rec);
        }

        stats.games++;
        stats.game_plies += total_plies;
        if (z == 1.0f) stats.white_wins++;
        else if (z == 0.0f) stats.black_wins++;
        else stats.draws++;
        switch (reason)
        {
            case EndReason::Checkmate: stats.end_checkmate++; break;
            case EndReason::Stalemate: stats.end_stalemate++; break;
            case EndReason::Insufficient: stats.end_insufficient++; break;
            case EndReason::Fifty: stats.end_fifty++; break;
            case EndReason::Repetition: stats.end_repetition++; break;
            case EndReason::MaxPlies: stats.end_max_plies++; break;
            case EndReason::Resign: stats.end_resign++; break;
        }
        if (slot.curriculum) stats.curriculum_games++;
        if (slot.playthrough)
        {
            stats.playthrough_games++;
            // A would-be resignation is a false positive if that side did not lose.
            if (slot.would_resign[0])
            {
                stats.playthrough_would_resign++;
                if (z != 0.0f) stats.playthrough_false_resign++;
            }
            if (slot.would_resign[1])
            {
                stats.playthrough_would_resign++;
                if (z != 1.0f) stats.playthrough_false_resign++;
            }
        }

        if (outfile) outfile.flush();
        if (config.save_pgn) write_pgn(slot, z, slot_index);
        reset_slot(slot);
    }

    // One move of one board: its search request, then the search's result applied to the board.
    struct MovePlan
    {
        chessrl::mcts::MctsOptions options;
        bool full{true};
        bool sampled{false};
        bool policy_move{false};
        int simulations{0};
    };

    MovePlan plan_move(BoardSlot& slot)
    {
        const SelfPlayConfig& cfg = config;
        MovePlan plan;
        chessrl::mcts::MctsOptions& opt = plan.options;
        // The search starts from the board itself (its state history gives repetitions) and the
        // history stack below, so it reads neither request.fen nor request.history.
        opt.request.board = slot.board;
        // Supply the AlphaZero history stack directly from the rolling
        // snapshots we already maintain (slot.past_planes), so the runner
        // skips the O(plies) build_root_history_stack replay it would
        // otherwise do on EVERY ply ⇒ O(plies^2)/game. Bit-identical to
        // build_root_history_stack: slot 0 = snapshot(slot.board) + its
        // isRepetition(1); slots 1..7 = past_planes[0..6] (each captured
        // pre-move with its repetition flag, newest-first — same ordering).
        {
            chessrl::mcts::HistoryStack hs{};
            hs[0] = chessrl::mcts::snapshot_piece_planes(slot.board);
            hs[0].repetition = slot.board.isRepetition(1);
            for (int s = 1; s < chessrl::mcts::kHistoryDepth; ++s)
            {
                const size_t k = static_cast<size_t>(s - 1);
                if (k < slot.past_planes.size())
                    hs[static_cast<size_t>(s)] = slot.past_planes[k];
            }
            opt.request.history_stack = hs;
        }

        // Playout-cap randomization: cheap searches play the game forward but
        // produce no training record and no root noise.
        const bool full = cfg.pcr_full_prob >= 1.0f || uniform01() < cfg.pcr_full_prob;
        const bool early = cfg.temperature_plies < 0 || slot.ply_count < cfg.temperature_plies;
        const float root_temp = early ? cfg.temperature_at_root : cfg.temperature_late;
        // A cheap move in the sampling phase can sample the raw policy instead: with
        // no simulations the search only expands the root, and summarize() samples
        // its priors; prior temperature 1 makes those softmax(logit), so the move is
        // drawn from softmax(logit / quick_policy_temperature).
        const bool policy_move = !full && early && root_temp > 1e-5F && cfg.quick_policy_temperature > 0.0f;
        plan.full = full;
        plan.policy_move = policy_move;
        plan.simulations = full ? cfg.simulations : (policy_move ? 0 : cfg.pcr_fast_simulations);
        plan.sampled = root_temp > 1e-5F;

        opt.simulations = plan.simulations;
        opt.cpuct = cfg.cpuct;
        opt.temperature = policy_move ? 1.0f : cfg.temperature;
        opt.temperature_at_root = policy_move ? cfg.quick_policy_temperature : root_temp;
        opt.virtual_loss_width = cfg.virtual_loss_width;
        opt.principal_variations = false;
        opt.use_dirichlet = full && cfg.use_dirichlet;
        opt.dirichlet_alpha = cfg.dirichlet_alpha;
        opt.dirichlet_epsilon = cfg.dirichlet_epsilon;
        opt.small_engine_path = cfg.engine_path;
        // Tree reuse: continue the subtree the previous search kept for the move played (not
        // for policy-sampled moves, which play from the priors without searching).
        if (cfg.tree_reuse && !policy_move)
        {
            opt.keep_tree = true;
            opt.reuse_tree = std::move(slot.tree);
        }
        slot.tree.reset();
        return plan;
    }

    void apply_move(BoardSlot& slot, size_t i, const MovePlan& plan, const chessrl::mcts::MctsSummary& summary,
                    SelfPlayMetrics& metrics)
    {
        const SelfPlayConfig& cfg = config;
        metrics.nodes_evaluated += summary.new_simulations;
        stats.simulations_run += summary.new_simulations;
        stats.reused_visits += summary.reused_visits;

        if (summary.action.empty())
        {
            reset_slot(slot);
            metrics.games_finished++;
            return;
        }

        if (plan.full) stats.full_searches++;
        else stats.fast_searches++;
        if (plan.policy_move) stats.policy_moves++;
        if (plan.full)
        {
            slot.pending_records.push_back(make_record(slot, summary, plan.simulations, true, plan.sampled));
            metrics.positions_generated++;
        }

        // Resignation: the side to move gives up once its root expected score
        // has stayed below the threshold for resign_consecutive of its moves.
        // Policy-sampled moves carry only the raw value head, not a search value,
        // so they neither extend nor reset the streak.
        const chess::Color stm = slot.board.sideToMove();
        const size_t colour = stm == chess::Color::WHITE ? 0 : 1;
        if (cfg.resign_threshold > 0.0f && !plan.policy_move)
        {
            const float expected = summary.value[0] + 0.5f * summary.value[1];
            slot.low_streak[colour] = expected < cfg.resign_threshold ? slot.low_streak[colour] + 1 : 0;
            if (slot.low_streak[colour] >= cfg.resign_consecutive)
            {
                if (slot.playthrough)
                {
                    slot.would_resign[colour] = true;
                }
                else
                {
                    finish_game(slot, i, colour == 0 ? 0.0f : 1.0f, EndReason::Resign);
                    metrics.games_finished++;
                    return;
                }
            }
        }

        // Apply move
        bool move_found = false;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, slot.board);
        for (const auto& m : moves)
        {
            if (chess::uci::moveToUci(m, slot.board.chess960()) == summary.action)
            {
                // Snapshot the pre-move position into the past_planes deque
                // BEFORE applying the move, so the next record sees this
                // position at slot 1 (t-1).
                chessrl::mcts::PiecePlanes pre = chessrl::mcts::snapshot_piece_planes(slot.board);
                pre.repetition = slot.board.isRepetition(1);
                slot.past_planes.push_front(pre);
                if (slot.past_planes.size() > 7) slot.past_planes.pop_back();

                slot.board.makeMove(m);
                slot.history.push_back(summary.action);
                slot.ply_count++;
                slot.tree = summary.tree;  // set only with tree_reuse
                move_found = true;
                break;
            }
        }

        if (!move_found)
        {
            std::cerr << "Move not found: " << summary.action << " in " << slot.board.getFen() << std::endl;
            reset_slot(slot);
            return;
        }

        // Check termination
        const auto [reason, result] = slot.board.isGameOver();
        if (result == chess::GameResult::NONE && slot.ply_count < cfg.max_plies)
        {
            return;
        }

        float z = 0.5f;
        const bool white_to_move = slot.board.sideToMove() == chess::Color::WHITE;
        if (result == chess::GameResult::LOSE) z = white_to_move ? 0.0f : 1.0f; // side to move is mated
        else if (result == chess::GameResult::WIN) z = white_to_move ? 1.0f : 0.0f;

        EndReason end = EndReason::MaxPlies;
        if (result != chess::GameResult::NONE)
        {
            switch (reason)
            {
                case chess::GameResultReason::CHECKMATE: end = EndReason::Checkmate; break;
                case chess::GameResultReason::STALEMATE: end = EndReason::Stalemate; break;
                case chess::GameResultReason::INSUFFICIENT_MATERIAL: end = EndReason::Insufficient; break;
                case chess::GameResultReason::FIFTY_MOVE_RULE: end = EndReason::Fifty; break;
                case chess::GameResultReason::THREEFOLD_REPETITION: end = EndReason::Repetition; break;
                default: end = EndReason::MaxPlies; break;
            }
        }
        finish_game(slot, i, z, end);
        metrics.games_finished++;
    }

    // ---- Asynchronous self-play (SelfPlayConfig::async_play). Player thread t plays boards t, t + n, ...
    // in turn, one move each, never waiting for the others. Preparing and applying a move run
    // under state_mutex (the record file, stats, the config and the opening RNG are shared); the
    // searches run outside it, each on the runner current when it starts (`runner` is read and
    // replaced atomically, so an engine swap does not wait for the searches in flight).
    std::mutex state_mutex;
    std::vector<std::thread> players;
    std::atomic<bool> stop_players{false};
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    size_t moves_done{0};  // since the last step() returned (progress_mutex)
    size_t wake_at{0};     // step() is waiting for this many moves
    SelfPlayMetrics progress;

    void start_players()
    {
        if (!players.empty()) return;
        const size_t n = std::min(slots.size(), static_cast<size_t>(config.threads > 0 ? config.threads : 256));
        for (size_t t = 0; t < n; ++t) players.emplace_back([this, t, n] { play(t, n); });
    }

    void play(size_t first, size_t stride)
    {
        while (!stop_players.load(std::memory_order_relaxed))
        {
            for (size_t i = first; i < slots.size() && !stop_players.load(std::memory_order_relaxed); i += stride)
            {
                MovePlan plan;
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    plan = plan_move(slots[i]);
                }
                const auto summary =
                    chessrl::mcts::run_small_mcts_shared(std::move(plan.options), std::atomic_load(&runner), cache);
                SelfPlayMetrics m;
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    apply_move(slots[i], i, plan, summary, m);
                }
                bool wake = false;
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    ++moves_done;
                    progress.games_finished += m.games_finished;
                    progress.positions_generated += m.positions_generated;
                    progress.nodes_evaluated += m.nodes_evaluated;
                    wake = wake_at > 0 && moves_done >= wake_at;
                }
                if (wake) progress_cv.notify_one();
            }
        }
    }

    ~Impl()
    {
        stop_players.store(true);
        for (auto& p : players) p.join();
    }

    Impl(const SelfPlayConfig& c) : config(c)
    {
        batch_config.engine_path = config.engine_path;
        batch_config.enable_telemetry = true;
        // Batch dispatch threshold. Sized to the worker count, not the
        // per-batch in-flight peak (which would be threads × width with
        // virtual loss). Measured: capping at ~threads keeps per-batch
        // latency ~1 ms; larger caps (≥256) push it past 5 ms and net-
        // regress total pos/sec because MCTS threads spend longer blocked
        // on ticket.get(). The 4 GPU workers pipeline well at this size.
        const int realistic_concurrency = (config.threads > 0)
            ? config.threads * searches_per_thread()
            : std::min(config.concurrency, 256);
        batch_config.max_batch_size = std::max(1, realistic_concurrency);
        runner = std::make_shared<chessrl::mcts::single::SmallBatchRunner>(batch_config);

        // hash_size <= 0 disables the TT entirely (skips lock+hash+lookup on
        // every leaf). At <2% hit rate the cache is net-negative, so leaving
        // this off is a real perf win for self-play with few transpositions.
        if (config.hash_size > 0)
        {
            cache = std::make_shared<chessrl::mcts::TranspositionTable>(config.hash_size);
        }

        slots.resize(config.concurrency);
        for (auto& slot : slots)
        {
            reset_slot(slot);
        }

        // std::fstream in|out doesn't create a missing file, so create it first.
        if (!std::filesystem::exists(config.output_path))
        {
            std::ofstream tmp(config.output_path);
        }

        // Truncate to a record boundary if a prior run left a partial trailing
        // record, so subsequent appends don't produce a misaligned record.
        {
            std::error_code ec;
            auto fsize = std::filesystem::file_size(config.output_path, ec);
            if (!ec)
            {
                size_t aligned = (fsize / kRecordSize) * kRecordSize;
                if (aligned != fsize)
                {
                    std::filesystem::resize_file(config.output_path, aligned, ec);
                }
            }
        }

        outfile.open(config.output_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!outfile)
        {
            std::cerr << "Failed to open output file: " << config.output_path << std::endl;
        }
        else
        {
            outfile.seekp(0, std::ios::end);
            size_t size = outfile.tellp();
            current_records = size / kRecordSize;
            // Resume from the existing sidecar if present so an interrupted
            // run reattaches the monotonic counter. While a FIFO buffer is
            // still filling, the counter must equal the record count so the
            // ring head stays at the oldest record.
            std::ifstream wf(config.output_path + ".writes");
            size_t prev = 0;
            if (wf) wf >> prev;
            total_writes = std::max(prev, current_records);
            if (config.fifo_buffer && current_records < config.max_buffer_size)
            {
                total_writes = current_records;
            }
        }
    }
};

SelfPlayWorker::SelfPlayWorker(const SelfPlayConfig& config)
    : impl_(new Impl(config))
{
}

SelfPlayWorker::~SelfPlayWorker()
{
    delete impl_;
}

SelfPlayMetrics SelfPlayWorker::step()
{
    SelfPlayMetrics metrics;
    auto start_time = std::chrono::high_resolution_clock::now();
    const SelfPlayConfig& cfg = impl_->config;
    const size_t n_slots = impl_->slots.size();

    if (cfg.async_play)
    {
        impl_->start_players();
        {
            std::unique_lock<std::mutex> lock(impl_->progress_mutex);
            impl_->wake_at = n_slots;
            impl_->progress_cv.wait(lock, [&] { return impl_->moves_done >= n_slots; });
            impl_->wake_at = 0;
            impl_->moves_done -= n_slots;
            metrics = impl_->progress;
            impl_->progress = SelfPlayMetrics{};
        }
        {
            std::lock_guard<std::mutex> lock(impl_->state_mutex);
            impl_->write_sidecars();
            chessrl::mcts::report_mcts_phases(cfg.threads);
        }
        metrics.runtime_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        return metrics;
    }

    // Prepare MCTS requests
    std::vector<chessrl::mcts::MctsOptions> options;
    options.reserve(n_slots);
    std::vector<Impl::MovePlan> plans(n_slots);
    for (size_t i = 0; i < n_slots; ++i)
    {
        plans[i] = impl_->plan_move(impl_->slots[i]);
        options.push_back(std::move(plans[i].options));
    }

    // Run MCTS
    double start_infer_ms = impl_->runner->statistics().total_inference_ms;
    const auto search_start = std::chrono::high_resolution_clock::now();
    auto summaries = chessrl::mcts::run_small_mcts_batch_shared(
        std::move(options), impl_->runner, impl_->cache, cfg.threads, searches_per_thread());
    double end_infer_ms = impl_->runner->statistics().total_inference_ms;

    auto current_time = std::chrono::high_resolution_clock::now();
    double mcts_duration_ms = std::chrono::duration<double, std::milli>(current_time - start_time).count();
    double infer_duration_ms = end_infer_ms - start_infer_ms;

    if (std::getenv("CHESSRL_PROFILE_LOG") != nullptr)
    {
        std::cerr << "[Profile] MCTS=" << mcts_duration_ms << "ms"
                  << " network=" << infer_duration_ms << "ms\n";
        if (impl_->cache) impl_->cache->print_stats();
        else std::cerr << "[Cache] disabled (hash_size=0)\n";
    }

    // Process results
    for (size_t i = 0; i < n_slots; ++i)
    {
        impl_->apply_move(impl_->slots[i], i, plans[i], summaries[i], metrics);
    }

    impl_->write_sidecars();

    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.runtime_seconds = std::chrono::duration<double>(end_time - start_time).count();
    // The serial parts of a step (the search threads have nothing to run during them), next to
    // the searches' wall time: preparing the requests, then applying the moves and writing.
    static const bool step_phases = std::getenv("CHESSRL_MCTS_PHASES") != nullptr;
    if (step_phases)
    {
        using ms = std::chrono::duration<double, std::milli>;
        std::cerr << "[step-phases] prep=" << ms(search_start - start_time).count()
                  << " search=" << ms(current_time - search_start).count()
                  << " post=" << ms(end_time - current_time).count() << "\n";
    }
    return metrics;
}

size_t SelfPlayWorker::get_buffer_size() const
{
    std::filesystem::path p(impl_->config.output_path);
    if (!std::filesystem::exists(p))
    {
        return 0;
    }

    size_t file_size = std::filesystem::file_size(p);
    return file_size / kRecordSize;
}

void SelfPlayWorker::set_engine(const std::string& engine_path)
{
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    impl_->config.engine_path = engine_path;
    impl_->batch_config.engine_path = engine_path;
    if (impl_->players.empty())
    {
        // Release the old engine's GPU memory before loading the new one.
        impl_->runner.reset();
        impl_->runner = std::make_shared<chessrl::mcts::single::SmallBatchRunner>(impl_->batch_config);
    }
    else
    {
        // Searches in flight finish on the old network, which goes with the last of them.
        std::atomic_store(&impl_->runner,
                          std::make_shared<chessrl::mcts::single::SmallBatchRunner>(impl_->batch_config));
    }
    impl_->stats.engine_swaps++;
}

void SelfPlayWorker::set_config(const SelfPlayConfig& config)
{
    SelfPlayConfig next = config;
    const SelfPlayConfig& cur = impl_->config;
    next.engine_path = cur.engine_path;
    next.output_path = cur.output_path;
    next.concurrency = cur.concurrency;
    next.threads = cur.threads;
    next.async_play = cur.async_play;
    next.hash_size = cur.hash_size;
    next.max_buffer_size = cur.max_buffer_size;
    next.fifo_buffer = cur.fifo_buffer;
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    impl_->config = next;
}

SelfPlayStats SelfPlayWorker::stats() const
{
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    return impl_->stats;
}

SelfPlayInferenceStats SelfPlayWorker::inference_stats() const
{
    SelfPlayInferenceStats out;
    if (const auto runner = std::atomic_load(&impl_->runner))
    {
        const auto rs = runner->statistics();
        out.batches = static_cast<int64_t>(rs.batches);
        out.positions_in_batches = static_cast<int64_t>(rs.positions);
        out.total_inference_ms = rs.total_inference_ms;
    }
    const auto et = chessrl::mcts::single::encoder_totals();
    out.encoder_calls = static_cast<int64_t>(et.calls);
    out.total_encoder_ms = static_cast<double>(et.nanos) / 1.0e6;
    return out;
}

} // namespace chessrl
