/* test_wfc_cpp.cpp — C++ wrapper tests with inline test framework */
#include <entropic/wfcpp.hpp>
#include <iostream>
#include <chrono>
#include <cassert>

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
    RUN(class_limits);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " passed\n";
    std::cout << "========================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
