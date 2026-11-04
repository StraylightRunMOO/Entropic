/* wfcpp.hpp — Entropic: C++17 Idiomatic Wrapper for wfc.h
 *
 * Thin RAII wrapper with type-safe enums, exception handling,
 * STL container integration, and modern C++ move semantics.
 *
 *   #include <wfcpp.hpp>
 *
 *   wfcpp::World world({64, 64, 1, 42});  // nx, ny, nz, seed
 *   world.add_pattern({0, 0, 1.0f});       // id, class_id, weight
 *   world.add_rule(0, 1, wfcpp::Dir::N);
 *   world.collapse();
 *   auto grid = world.export_grid();
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <functional>
#include <optional>
#include <tuple>

#define WFC_IMPLEMENTATION
#include "wfc.h"

namespace wfcpp {

/* ============================================================
 * Exception types
 * ============================================================ */
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

class InvalidArgument : public Error {
public:
    explicit InvalidArgument(const std::string& msg) : Error("InvalidArgument: " + msg) {}
};

class OutOfMemory : public Error {
public:
    explicit OutOfMemory(const std::string& msg) : Error("OutOfMemory: " + msg) {}
};

class Contradiction : public Error {
public:
    explicit Contradiction(const std::string& msg) : Error("Contradiction: " + msg) {}
    std::tuple<uint32_t, uint32_t, uint32_t> where() const { return pos; }
private:
    friend class World;
    std::tuple<uint32_t, uint32_t, uint32_t> pos{0, 0, 0};
};

class Uninitialized : public Error {
public:
    explicit Uninitialized(const std::string& msg) : Error("Uninitialized: " + msg) {}
};

/* ============================================================
 * Type-safe direction enum
 * ============================================================ */
enum class Dir {
    N = WFC_DIR_N,
    S = WFC_DIR_S,
    E = WFC_DIR_E,
    W = WFC_DIR_W,
    U = WFC_DIR_U,
    D = WFC_DIR_D
};

static inline wfc_dir_t to_c_dir(Dir d) {
    return static_cast<wfc_dir_t>(d);
}

static inline Dir opposite(Dir d) {
    switch (d) {
        case Dir::N: return Dir::S;
        case Dir::S: return Dir::N;
        case Dir::E: return Dir::W;
        case Dir::W: return Dir::E;
        case Dir::U: return Dir::D;
        case Dir::D: return Dir::U;
    }
    return Dir::N; /* unreachable */
}

/* ============================================================
 * Pattern definition builder
 * ============================================================ */
struct Pattern {
    uint16_t id = 0;
    uint16_t class_id = 0;
    float weight = 1.0f;
    uint32_t freq = 0;
    uint8_t num_layers = 0;
    std::vector<uint16_t> layer_types;

    Pattern() = default;
    Pattern(uint16_t id_, uint16_t class_id_, float weight_ = 1.0f)
        : id(id_), class_id(class_id_), weight(weight_) {}

    wfc_pattern_def_t to_c() const {
        wfc_pattern_def_t p{};
        p.id = id;
        p.class_id = class_id;
        p.weight = weight;
        p.freq = freq;
        p.num_layers = num_layers;
        for (size_t i = 0; i < layer_types.size() && i < WFC_MAX_LAYERS; ++i)
            p.layer_types[i] = layer_types[i];
        return p;
    }
};

/* ============================================================
 * Configuration
 * ============================================================ */
struct Config {
    uint32_t nx = 64;
    uint32_t ny = 64;
    uint32_t nz = 1;
    uint32_t seed = 0;
    bool enable_backtracking = false;
    bool enable_spectral = false;
    bool enable_copula = false;
    bool seamless = false;
    float spectral_weight = 1.0f;
    float copula_weight = 1.0f;

    wfc_config_t to_c() const {
        wfc_config_t c{};
        c.nx = nx;
        c.ny = ny;
        c.nz = nz;
        c.seed = seed;
        c.enable_backtracking = enable_backtracking ? 1 : 0;
        c.enable_spectral = enable_spectral ? 1 : 0;
        c.enable_copula = enable_copula ? 1 : 0;
        c.seamless = seamless ? 1 : 0;
        c.spectral_weight = spectral_weight;
        c.copula_weight = copula_weight;
        return c;
    }
};

/* ============================================================
 * Cell view (read-only accessor)
 * ============================================================ */
class CellView {
public:
    CellView() = default;
    CellView(const wfc_cell_t* cell, uint32_t idx) : cell_(cell), idx_(idx) {}

    bool collapsed() const { return cell_ && cell_->collapsed != WFC_PATTERN_INVALID; }
    uint16_t pattern_id() const { return cell_ ? cell_->collapsed : WFC_PATTERN_INVALID; }
    float entropy() const { return cell_ ? cell_->entropy : 0.0f; }
    uint32_t index() const { return idx_; }

private:
    const wfc_cell_t* cell_ = nullptr;
    uint32_t idx_ = 0;
};

/* ============================================================
 * Main World class (RAII wrapper)
 * ============================================================ */
class World {
public:
    /* -- Constructors / Destructor -- */
    World() = default;

    explicit World(const Config& cfg) {
        auto c = cfg.to_c();
        world_ = wfc_world_create(&c);
        if (!world_) throw OutOfMemory("Failed to create WFC world");
    }

    ~World() {
        if (world_) wfc_world_destroy(world_);
    }

    /* -- Move semantics -- */
    World(World&& other) noexcept : world_(other.world_) {
        other.world_ = nullptr;
    }
    World& operator=(World&& other) noexcept {
        if (this != &other) {
            if (world_) wfc_world_destroy(world_);
            world_ = other.world_;
            other.world_ = nullptr;
        }
        return *this;
    }

    /* No copy */
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    /* -- Pattern / Rule management -- */
    void add_pattern(const Pattern& pat) {
        check();
        auto c = pat.to_c();
        wfc_err_t err = wfc_add_pattern(world_, &c);
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid pattern ID or world full");
        if (err != WFC_OK) throw Error("Failed to add pattern");
    }

    void add_rule(uint16_t pat_a, uint16_t pat_b, Dir dir) {
        check();
        wfc_err_t err = wfc_add_rule(world_, pat_a, pat_b, to_c_dir(dir));
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid rule parameters");
        if (err != WFC_OK) throw Error("Failed to add rule");
    }

    void add_symmetric_rule(uint16_t pat_a, uint16_t pat_b, Dir dir) {
        add_rule(pat_a, pat_b, dir);
        add_rule(pat_b, pat_a, opposite(dir));
    }

    /* Pin a cell before collapse. Weight 0 is legal and will not be chosen
     * in any other cell. */
    void fix_cell(uint32_t x, uint32_t y, uint32_t z, uint16_t pat) {
        check();
        wfc_err_t err = wfc_fix_cell(world_, x, y, z, pat);
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid cell or pattern");
        if (err != WFC_OK) throw Error("Failed to fix cell");
    }

    /* -- Class constraints -- */
    void set_class_limit(uint16_t class_id, uint32_t min_count, uint32_t max_count) {
        check();
        wfc_err_t err = wfc_set_class_limit(world_, class_id, min_count, max_count);
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid class ID");
    }

    /* -- Learning from sample -- */
    void learn_from_sample(const std::vector<uint16_t>& sample, uint32_t sx, uint32_t sy, uint32_t sz = 1) {
        check();
        if (sample.size() < sx * sy * sz) throw InvalidArgument("Sample buffer too small");
        wfc_err_t err = wfc_learn_from_sample(world_, sample.data(), sx, sy, sz);
        if (err != WFC_OK) throw Error("Failed to learn from sample");
    }

    /* -- Spectral fields -- */
#ifdef WFC_USE_FAF
    void set_gumbel_field(double H, uint64_t noise_seed) {
        check();
        wfc_err_t err = wfc_set_gumbel_field(world_, H, noise_seed);
        if (err == WFC_EUNINIT) throw Uninitialized("Spectral not enabled in config");
        if (err == WFC_ENOMEM) throw OutOfMemory("Failed to allocate spectral field");
        if (err != WFC_OK) throw Error("Failed to set Gumbel field");
    }

    void set_matern_field(double kappa, double nu, uint64_t noise_seed) {
        check();
        wfc_err_t err = wfc_set_matern_field(world_, kappa, nu, noise_seed);
        if (err == WFC_EUNINIT) throw Uninitialized("Spectral not enabled in config");
        if (err == WFC_ENOMEM) throw OutOfMemory("Failed to allocate spectral field");
        if (err != WFC_OK) throw Error("Failed to set Matérn field");
    }
#endif

    /* -- Multiscale hierarchy -- */
#ifdef WFC_ENABLE_HIERARCHY
    void hierarchy_init(uint32_t nlevels) {
        check();
        wfc_err_t err = wfc_hierarchy_init(world_, nlevels);
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid hierarchy nlevels");
        if (err == WFC_ENOMEM) throw OutOfMemory("Failed to allocate hierarchy");
        if (err != WFC_OK) throw Error("Failed to init hierarchy");
    }

    void hierarchy_set_wavelet(int family, uint32_t dwt_levels) {
        check();
        wfc_err_t err = wfc_hierarchy_set_wavelet(world_, family, dwt_levels);
        if (err == WFC_EUNINIT) throw Uninitialized("Hierarchy not initialised");
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid wavelet family");
        if (err != WFC_OK) throw Error("Failed to set wavelet");
    }

    void hierarchy_set_target(const float* target, uint32_t count) {
        check();
        wfc_err_t err = wfc_hierarchy_set_target(world_, target, count);
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid hierarchy target");
        if (err == WFC_ENOMEM) throw OutOfMemory("Failed to allocate hierarchy target");
        if (err != WFC_OK) throw Error("Failed to set target");
    }

    void hierarchy_set_target(const std::vector<float>& target) {
        hierarchy_set_target(target.data(), static_cast<uint32_t>(target.size()));
    }

    void hierarchy_set_mode(int mode) {
        check();
        wfc_err_t err = wfc_hierarchy_set_mode(world_, mode);
        if (err == WFC_EUNINIT) throw Uninitialized("Hierarchy not initialised");
        if (err == WFC_EINVAL) throw InvalidArgument("Invalid hierarchy mode");
        if (err != WFC_OK) throw Error("Failed to set hierarchy mode");
    }

    /* Decompose the live fBm/Matérn field once into the detail buffers. */
    void hierarchy_adopt_spectral() {
        check();
        wfc_err_t err = wfc_hierarchy_adopt_spectral(world_);
        if (err == WFC_EUNINIT) throw Uninitialized("Hierarchy or spectral field is not ready");
        if (err == WFC_ENOMEM) throw OutOfMemory("Failed to adopt spectral field");
        if (err != WFC_OK) throw Error("Failed to adopt spectral field");
    }

    float hierarchy_residual_energy(uint32_t level) const {
        check();
        return wfc_hierarchy_residual_energy(world_, level);
    }

    void hierarchy_collapse() {
        check();
        wfc_err_t err = wfc_hierarchy_collapse(world_);
        if (err == WFC_ECONTRADICTION) throw Contradiction("hierarchy collapse");
        if (err != WFC_OK) throw Error("Hierarchy collapse failed");
    }
#endif

    /* -- Coroutine parallelism -- */
#ifdef WFC_ENABLE_SUSPENDERS
    void parallel_init(unsigned num_workers = 0) {
        check();
        wfc_err_t err = wfc_parallel_init(world_, num_workers);
        if (err == WFC_ENOMEM) throw OutOfMemory("Failed to init parallel runtime");
        if (err != WFC_OK) throw Error("Failed to init parallel runtime");
    }

    void parallel_shutdown() {
        check();
        wfc_parallel_shutdown(world_);
    }
#endif

    /* -- Contradiction callback -- */
    using ContradictionCallback = std::function<void(uint32_t x, uint32_t y, uint32_t z)>;

    void set_contradiction_callback(ContradictionCallback cb) {
        check();
        cb_ = std::move(cb);
        if (cb_) {
            wfc_set_contradiction_callback(world_, &World::static_contradiction_cb, this);
        } else {
            wfc_set_contradiction_callback(world_, nullptr, nullptr);
        }
    }

    /* -- Solve: full collapse -- */
    void collapse() {
        check();
        wfc_err_t err = wfc_collapse(world_);
        if (err == WFC_ECONTRADICTION) {
            Contradiction ex("Wave function contradiction");
            ex.pos = {last_contra_x_, last_contra_y_, last_contra_z_};
            throw ex;
        }
        if (err != WFC_OK) throw Error("Collapse failed");
    }

    /* -- Step-by-step execution -- */
    void step_init() {
        check();
        wfc_err_t err = wfc_step_init(world_);
        if (err != WFC_OK) throw Error("Step init failed");
    }

    bool step() {
        check();
        wfc_err_t err = wfc_step(world_);
        if (err == WFC_ECONTRADICTION) {
            Contradiction ex("Wave function contradiction");
            ex.pos = {last_contra_x_, last_contra_y_, last_contra_z_};
            throw ex;
        }
        if (err != WFC_OK) throw Error("Step failed");
        return !wfc_step_done(world_);
    }

    bool step_done() const {
        check();
        return wfc_step_done(world_);
    }

    /* -- Query results -- */
    uint16_t get_pattern(uint32_t x, uint32_t y, uint32_t z = 0) const {
        check();
        return wfc_get_cell_pattern(world_, x, y, z);
    }

    const wfc_pattern_def_t* get_pattern_def(uint16_t pat_id) const {
        check();
        return wfc_get_pattern(world_, pat_id);
    }

    /* Logical linear index: z * ny * nx + y * nx + x, not the padded slot. */
    CellView cell(uint32_t idx) const {
        check();
        uint32_t logical = world_->nx * world_->ny * world_->nz;
        if (idx >= logical) throw InvalidArgument("Cell index out of range");
        uint32_t x = idx % world_->nx;
        uint32_t y = (idx / world_->nx) % world_->ny;
        uint32_t z = idx / (world_->nx * world_->ny);
        return CellView(&world_->cells[wfc_idx(world_, x, y, z)], idx);
    }

    CellView cell_at(uint32_t x, uint32_t y, uint32_t z = 0) const {
        check();
        if (x >= world_->nx || y >= world_->ny || z >= world_->nz)
            throw InvalidArgument("Cell coordinates out of range");
        uint32_t lin = (z * world_->ny + y) * world_->nx + x;
        return CellView(&world_->cells[wfc_idx(world_, x, y, z)], lin);
    }

    /* -- Export grid -- */
    std::vector<uint16_t> export_grid() const {
        check();
        uint32_t total = world_->nx * world_->ny * world_->nz;
        std::vector<uint16_t> grid(total);
        wfc_err_t err = wfc_export_grid(world_, grid.data(), grid.size());
        if (err != WFC_OK) throw Error("Export failed");
        return grid;
    }

    /* -- Reset for re-solving -- */
    void reset() {
        check();
        wfc_world_reset(world_);
    }

    /* -- Accessors -- */
    uint32_t nx() const { check(); return world_->nx; }
    uint32_t ny() const { check(); return world_->ny; }
    uint32_t nz() const { check(); return world_->nz; }
    uint32_t num_cells() const { check(); return world_->num_cells; }
    uint32_t num_patterns() const { check(); return world_->num_patterns; }
    uint32_t num_classes() const { check(); return world_->num_classes; }
    bool is_3d() const { check(); return world_->is_3d != 0; }
    bool seamless() const { check(); return world_->seamless != 0; }
    uint64_t rng_state() const { check(); return world_->rng_state; }

    /* -- Direct C access (escape hatch) -- */
    wfc_world_t* c_ptr() { return world_; }
    const wfc_world_t* c_ptr() const { return world_; }

private:
    wfc_world_t* world_ = nullptr;
    ContradictionCallback cb_;
    uint32_t last_contra_x_ = 0, last_contra_y_ = 0, last_contra_z_ = 0;

    void check() const {
        if (!world_) throw Uninitialized("World is null or has been moved");
    }

    static void static_contradiction_cb(uint32_t x, uint32_t y, uint32_t z, void* user_data) {
        auto* self = static_cast<World*>(user_data);
        self->last_contra_x_ = x;
        self->last_contra_y_ = y;
        self->last_contra_z_ = z;
        if (self->cb_) self->cb_(x, y, z);
    }
};

/* ============================================================
 * Utility: Grid coordinate conversion
 * ============================================================ */
struct GridCoords {
    uint32_t x, y, z;
};

static inline GridCoords idx_to_coords(const World& w, uint32_t idx) {
    uint32_t nx = w.nx();
    uint32_t ny = w.ny();
    uint32_t x = idx % nx;
    uint32_t y = (idx / nx) % ny;
    uint32_t z = idx / (nx * ny);
    return {x, y, z};
}

static inline uint32_t coords_to_idx(const World& w, uint32_t x, uint32_t y, uint32_t z) {
    return (z * w.ny() + y) * w.nx() + x;
}

/* ============================================================
 * Utility: Pretty-print grid to string
 * ============================================================ */
static inline std::string grid_to_string(const World& w, const std::vector<uint16_t>& grid) {
    std::string s;
    for (uint32_t y = 0; y < w.ny(); ++y) {
        for (uint32_t x = 0; x < w.nx(); ++x) {
            uint16_t p = grid[y * w.nx() + x];
            if (p == WFC_PATTERN_INVALID) s += '?';
            else if (p < 10) s += static_cast<char>('0' + p);
            else if (p < 36) s += static_cast<char>('A' + p - 10);
            else s += '#';
        }
        s += '\n';
    }
    return s;
}

} /* namespace wfcpp */
