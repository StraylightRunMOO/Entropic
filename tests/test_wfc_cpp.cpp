/* test_wfc_cpp.cpp — C++ wrapper tests with inline test framework */
#undef NDEBUG
#include <entropic/wfcpp.hpp>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cstdint>

using namespace wfcpp;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) void test_##name()
#define RUN(name) do { \
    test_count++; \
    std::cout << "  [" << test_count << "] " #name " ... "; \
    try { test_##name(); pass_count++; std::cout << "PASS\n"; } \
    catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << "\n"; } \
} while(0)

TEST(create_destroy) {
    Config cfg{.nx = 8, .ny = 8, .nz = 1, .seed = 42};
    World w(cfg);
    assert(w.nx() == 8);
    assert(w.ny() == 8);
    assert(w.nz() == 1);
    assert(w.num_cells() == 64);
    assert(!w.is_3d());
}

TEST(move_semantics) {
    Config cfg{.nx = 4, .ny = 4, .seed = 42};
    World w1(cfg);
    World w2(std::move(w1));
    assert(w2.nx() == 4);
}

TEST(add_pattern) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 0, 1.0f));
    assert(w.num_patterns() == 2);
}

TEST(invalid_pattern_throws) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    Pattern bad{256, 0, 1.0f};
    try {
        w.add_pattern(bad);
        assert(false && "Should have thrown");
    } catch (const InvalidArgument&) {
        // expected
    }
}

TEST(add_rule) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    w.add_rule(0, 1, Dir::E);
}

TEST(collapse_1x1) {
    World w({.nx = 1, .ny = 1, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.collapse();
    assert(w.get_pattern(0, 0) == 0);
}

TEST(collapse_2x2) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
        w.add_rule(1, 1, d);
    }
    w.collapse();
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++)
            assert(w.get_pattern(x, y) == 0 || w.get_pattern(x, y) == 1);
}

TEST(export_grid) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    auto grid = w.export_grid();
    assert(grid.size() == 4);
    for (auto p : grid) assert(p == 0);
}

TEST(reset) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    assert(w.get_pattern(0, 0) == 0);
    w.reset();
    assert(w.num_patterns() == 1);
}

TEST(step_by_step) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.step_init();
    int count = 0;
    while (!w.step_done()) {
        w.step();
        count++;
        assert(count < 100);
    }
    assert(count == 4);
}

TEST(cell_view) {
    World w({.nx = 1, .ny = 1, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.collapse();
    auto cell = w.cell(0);
    assert(cell.collapsed());
    assert(cell.pattern_id() == 0);
}

TEST(seamless_mode) {
    World w({.nx = 4, .ny = 4, .seed = 42, .seamless = true});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
}

TEST(grid_to_string) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    auto grid = w.export_grid();
    auto s = grid_to_string(w, grid);
    assert(!s.empty());
}

TEST(config_defaults) {
    Config cfg;
    assert(cfg.nx == 64);
    assert(cfg.ny == 64);
    assert(cfg.nz == 1);
    assert(!cfg.seamless);
}

TEST(pattern_defaults) {
    Pattern p;
    assert(p.weight == 1.0f);
}

TEST(pattern_builder) {
    Pattern p(5, 2, 0.5f);
    assert(p.id == 5);
    assert(p.class_id == 2);
    assert(p.weight == 0.5f);
}

TEST(dir_opposite) {
    assert(opposite(Dir::N) == Dir::S);
    assert(opposite(Dir::S) == Dir::N);
    assert(opposite(Dir::E) == Dir::W);
    assert(opposite(Dir::W) == Dir::E);
    assert(opposite(Dir::U) == Dir::D);
    assert(opposite(Dir::D) == Dir::U);
}

TEST(learn_from_sample) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    std::vector<uint16_t> sample = {
        0,1,0,1,
        1,0,1,0,
        0,1,0,1,
        1,0,1,0
    };
    w.learn_from_sample(sample, 4, 4, 1);
    w.collapse();
}

TEST(large_collapse) {
    World w({.nx = 32, .ny = 32, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(1, 1, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
    }
    w.collapse();
    auto grid = w.export_grid();
    assert(grid.size() == 1024);
}

TEST(contradiction_callback) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    bool fired = false;
    w.set_contradiction_callback([&](uint32_t x, uint32_t y, uint32_t z) {
        (void)x; (void)y; (void)z;
        fired = true;
    });
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    assert(!fired);
}

TEST(multiple_collapses) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 0, 1.0f));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(1, 1, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
    }
    for (int i = 0; i < 5; i++) {
        w.collapse();
        auto grid = w.export_grid();
        for (auto p : grid) assert(p == 0 || p == 1);
        w.reset();
    }
}

TEST(coords_conversion) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    uint32_t idx = coords_to_idx(w, 2, 3, 0);
    assert(idx == 14);
    auto c = idx_to_coords(w, 14);
    assert(c.x == 2);
    assert(c.y == 3);
    assert(c.z == 0);
}

TEST(performance_small) {
    auto t0 = std::chrono::steady_clock::now();
    World w({.nx = 16, .ny = 16, .seed = 42});
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 0, 1.0f));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(1, 1, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
    }
    w.collapse();
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    assert(ms < 100.0);
}

TEST(pattern_with_layers) {
    Pattern p(0, 0, 1.0f);
    p.num_layers = 2;
    p.layer_types = {1, 2};
    auto c = p.to_c();
    assert(c.num_layers == 2);
    assert(c.layer_types[0] == 1);
    assert(c.layer_types[1] == 2);
}

TEST(config_round_trip) {
    Config cfg{.nx = 16, .ny = 32, .nz = 2, .seed = 12345,
               .enable_backtracking = true,
               .seamless = true,
               .spectral_weight = 0.5f,
               .copula_weight = 0.3f};
    auto c = cfg.to_c();
    assert(c.nx == 16);
    assert(c.ny == 32);
    assert(c.nz == 2);
    assert(c.seed == 12345);
    assert(c.enable_backtracking == 1);
    assert(c.seamless == 1);
    assert(c.spectral_weight == 0.5f);
    assert(c.copula_weight == 0.3f);
}

TEST(null_callback) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.set_contradiction_callback(nullptr);
    w.collapse();
}

#ifdef WFC_ENABLE_HIERARCHY
TEST(hierarchy_collapse) {
    Config cfg{.nx = 16, .ny = 16, .seed = 42, .enable_spectral = true};
    World w(cfg);
    w.add_pattern(Pattern(0, 0, 5.0f));
    w.add_pattern(Pattern(1, 1, 3.0f));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
        w.add_rule(1, 1, d);
    }
    w.hierarchy_init(2);
    w.hierarchy_set_wavelet(FAF_WAVELET_CDF97, 0);
    std::vector<float> target(w.num_cells());
    for (size_t i = 0; i < target.size(); ++i) target[i] = (float)i;
    w.hierarchy_set_target(target);
    w.hierarchy_collapse();
    auto grid = w.export_grid();
    for (auto p : grid) assert(p == 0 || p == 1);
}

TEST(hierarchy_invalid_nlevels) {
    World w({.nx = 16, .ny = 16, .seed = 42, .enable_spectral = true});
    try {
        w.hierarchy_init(1);
        assert(false && "should have thrown");
    } catch (const InvalidArgument&) {}
    try {
        w.hierarchy_init(9);
        assert(false && "should have thrown");
    } catch (const InvalidArgument&) {}
}
#endif

TEST(get_pattern_def_and_symmetric) {
    World w({.nx = 4, .ny = 4, .seed = 1});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    w.add_symmetric_rule(0, 1, Dir::E);
    const wfc_pattern_def_t* def = w.get_pattern_def(0);
    assert(def && def->id == 0);
    assert(w.num_classes() >= 1);
    assert(w.c_ptr() != nullptr);
}

TEST(uninitialized_after_move) {
    World w1({.nx = 2, .ny = 2, .seed = 1});
    World w2(std::move(w1));
    try {
        (void)w1.nx();
        assert(false && "should throw");
    } catch (const Uninitialized&) {}
    World empty;
    try {
        empty.collapse();
        assert(false && "should throw");
    } catch (const Uninitialized&) {}
}

TEST(three_d_export) {
    World w({.nx = 2, .ny = 2, .nz = 2, .seed = 4});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W, Dir::U, Dir::D})
        w.add_rule(0, 0, d);
    w.collapse();
    auto grid = w.export_grid();
    assert(grid.size() == 8);
    assert(w.is_3d());
    assert(w.get_pattern(0, 0, 1) == 0);
}

TEST(contradiction_throws) {
    World w({.nx = 2, .ny = 2, .seed = 1});
    w.add_pattern(Pattern(0, 0));
    try {
        w.collapse();
        assert(false && "should throw");
    } catch (const Contradiction& e) {
        auto [x, y, z] = e.where();
        (void)x; (void)y; (void)z;
    }
}

#if defined(WFC_ENABLE_BACKTRACK)
TEST(backtrack_enabled) {
    Config cfg{.nx = 4, .ny = 4, .seed = 13, .enable_backtracking = true};
    World w(cfg);
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 0, 1.0f));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(1, 1, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
    }
    w.collapse();
    auto grid = w.export_grid();
    for (auto p : grid) assert(p == 0 || p == 1);
}
#endif

#if defined(WFC_ENABLE_COPULA)
TEST(copula_via_c_ptr) {
    Config cfg{.nx = 4, .ny = 4, .seed = 2, .enable_copula = true, .copula_weight = 1.0f};
    World w(cfg);
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 1));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(1, 1, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
    }
    wfc_world_t* raw = w.c_ptr();
    raw->copula_z[0] = 0.0f;
    raw->copula_z[1] = 0.0f;
    for (int d = 0; d < 4; d++)
        raw->copula_rho[d][0][0] = 0.7f;
    w.collapse();
}
#endif

#if defined(WFC_USE_FAF)
TEST(spectral_fields) {
    Config cfg{.nx = 8, .ny = 8, .seed = 3, .enable_spectral = true};
    World w(cfg);
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
        w.add_rule(1, 1, d);
    }
    w.set_gumbel_field(0.6, 11);
    w.collapse();
    w.reset();
    w.set_matern_field(1.0, 1.5, 12);
    w.collapse();
}

TEST(spectral_uninitialized) {
    World w({.nx = 4, .ny = 4, .seed = 1});
    try {
        w.set_gumbel_field(0.5, 1);
        assert(false && "should throw");
    } catch (const Uninitialized&) {}
}
#endif

TEST(invalid_rule_throws) {
    World w({.nx = 2, .ny = 2, .seed = 1});
    w.add_pattern(Pattern(0, 0));
    try {
        w.add_rule(0, 0, static_cast<Dir>(99));
        assert(false && "should throw");
    } catch (const InvalidArgument&) {}
}

TEST(learn_too_small) {
    World w({.nx = 2, .ny = 2, .seed = 1});
    w.add_pattern(Pattern(0, 0));
    std::vector<uint16_t> tiny = {0};
    try {
        w.learn_from_sample(tiny, 2, 2, 1);
        assert(false && "should throw");
    } catch (const InvalidArgument&) {}
}

TEST(cell_oob) {
    World w({.nx = 2, .ny = 2, .seed = 1});
    try {
        (void)w.cell(999);
        assert(false && "should throw");
    } catch (const InvalidArgument&) {}
}

TEST(logical_cell_on_padded_grid) {
    World w({.nx = 3, .ny = 3, .seed = 4});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    assert(w.num_cells() == 16);
    try {
        (void)w.cell(9);
        assert(false && "logical OOB should throw");
    } catch (const InvalidArgument&) {}
    auto cv = w.cell_at(2, 2);
    assert(cv.collapsed());
    assert(cv.pattern_id() == w.get_pattern(2, 2));
    assert(cv.index() == 8);
    assert((reinterpret_cast<uintptr_t>(w.c_ptr()->cells) % 64u) == 0);
}

TEST(class_limits) {
    World w({.nx = 8, .ny = 8, .seed = 42});
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 1, 1.0f));
    w.add_pattern(Pattern(2, 1, 0.5f));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                w.add_rule(i, j, d);
    }
    w.set_class_limit(1, 0, 4);
    w.collapse();
    auto grid = w.export_grid();
    int count_class1 = 0;
    for (auto p : grid)
        if (p == 1 || p == 2) count_class1++;
    assert(count_class1 <= 4);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "WFC Engine C++ Test Suite\n";
    std::cout << "========================================\n\n";

    RUN(create_destroy);
    RUN(move_semantics);
    RUN(add_pattern);
    RUN(invalid_pattern_throws);
    RUN(add_rule);
    RUN(collapse_1x1);
    RUN(collapse_2x2);
    RUN(export_grid);
    RUN(reset);
    RUN(step_by_step);
    RUN(cell_view);
    RUN(seamless_mode);
    RUN(grid_to_string);
    RUN(config_defaults);
    RUN(pattern_defaults);
    RUN(pattern_builder);
    RUN(dir_opposite);
    RUN(learn_from_sample);
    RUN(large_collapse);
    RUN(contradiction_callback);
    RUN(multiple_collapses);
    RUN(coords_conversion);
    RUN(performance_small);
    RUN(pattern_with_layers);
    RUN(config_round_trip);
    RUN(null_callback);
#ifdef WFC_ENABLE_HIERARCHY
    RUN(hierarchy_collapse);
    RUN(hierarchy_invalid_nlevels);
#endif
    RUN(get_pattern_def_and_symmetric);
    RUN(uninitialized_after_move);
    RUN(three_d_export);
    RUN(contradiction_throws);
#if defined(WFC_ENABLE_BACKTRACK)
    RUN(backtrack_enabled);
#endif
#if defined(WFC_ENABLE_COPULA)
    RUN(copula_via_c_ptr);
#endif
#if defined(WFC_USE_FAF)
    RUN(spectral_fields);
    RUN(spectral_uninitialized);
#endif
    RUN(invalid_rule_throws);
    RUN(learn_too_small);
    RUN(cell_oob);
    RUN(logical_cell_on_padded_grid);
    RUN(class_limits);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " passed\n";
    std::cout << "========================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
