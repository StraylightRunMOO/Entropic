/* test_wfc_basic.cpp — C++ wrapper basic tests using Google Test */
#include <gtest/gtest.h>
#include <entropic/wfcpp.hpp>
#include <cstdint>

using namespace wfcpp;

TEST(WorldTest, CreateDestroy) {
    Config cfg{.nx = 8, .ny = 8, .nz = 1, .seed = 42};
    World w(cfg);
    EXPECT_EQ(w.nx(), 8);
    EXPECT_EQ(w.ny(), 8);
    EXPECT_EQ(w.nz(), 1);
    EXPECT_EQ(w.num_cells(), 64);
    EXPECT_FALSE(w.is_3d());
}

TEST(WorldTest, MoveSemantics) {
    Config cfg{.nx = 4, .ny = 4, .seed = 42};
    World w1(cfg);
    World w2(std::move(w1));
    EXPECT_EQ(w2.nx(), 4);
}

TEST(WorldTest, AddPattern) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 0, 1.0f));
    EXPECT_EQ(w.num_patterns(), 2);
}

TEST(WorldTest, InvalidPatternThrows) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    Pattern bad{256, 0, 1.0f};
    EXPECT_THROW(w.add_pattern(bad), InvalidArgument);
}

TEST(WorldTest, AddRule) {
    World w({.nx = 4, .ny = 4, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    w.add_rule(0, 1, Dir::E);
}

TEST(WorldTest, Collapse1x1) {
    World w({.nx = 1, .ny = 1, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.collapse();
    EXPECT_EQ(w.get_pattern(0, 0), 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(w.c_ptr()->cells) % 64u, 0u);
    auto cv = w.cell(0);
    EXPECT_TRUE(cv.collapsed());
    EXPECT_EQ(cv.pattern_id(), 0);
}

TEST(WorldTest, Collapse2x2) {
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
            EXPECT_TRUE(w.get_pattern(x, y) == 0 || w.get_pattern(x, y) == 1);
}

TEST(WorldTest, ExportGrid) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    auto grid = w.export_grid();
    EXPECT_EQ(grid.size(), 4);
    for (auto p : grid) EXPECT_EQ(p, 0);
}

TEST(WorldTest, Reset) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    EXPECT_EQ(w.get_pattern(0, 0), 0);
    w.reset();
    EXPECT_EQ(w.num_patterns(), 1);
}

TEST(WorldTest, StepByStep) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.step_init();
    int count = 0;
    while (w.step()) {
        count++;
        ASSERT_LT(count, 100);
    }
    EXPECT_GT(count, 0);
    EXPECT_TRUE(w.step_done());
}

TEST(WorldTest, CellView) {
    World w({.nx = 1, .ny = 1, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.collapse();
    auto cell = w.cell(0);
    EXPECT_TRUE(cell.collapsed());
    EXPECT_EQ(cell.pattern_id(), 0);
}

TEST(WorldTest, LogicalCellIndexOnPaddedGrid) {
    /* 3x3 allocates 4x4; cell(i) is logical, not the padded slot. */
    World w({.nx = 3, .ny = 3, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    EXPECT_EQ(w.num_cells(), 16u);
    EXPECT_THROW(w.cell(9), InvalidArgument);
    auto c8 = w.cell(8);
    EXPECT_TRUE(c8.collapsed());
    EXPECT_EQ(c8.pattern_id(), w.get_pattern(2, 2));
    auto at = w.cell_at(2, 2);
    EXPECT_EQ(at.pattern_id(), w.get_pattern(2, 2));
    EXPECT_EQ(at.index(), 8u);
}

TEST(WorldTest, SeamlessMode) {
    World w({.nx = 4, .ny = 4, .seed = 42, .seamless = true});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
}

TEST(WorldTest, GridToString) {
    World w({.nx = 2, .ny = 2, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.collapse();
    auto grid = w.export_grid();
    auto s = grid_to_string(w, grid);
    EXPECT_FALSE(s.empty());
}

#ifdef WFC_ENABLE_HIERARCHY
TEST(WorldTest, HierarchyCollapse) {
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
    std::vector<float> target(w.num_cells(), 1.0f);
    w.hierarchy_set_target(target);
    w.hierarchy_collapse();
    auto grid = w.export_grid();
    for (auto p : grid)
        EXPECT_TRUE(p == 0 || p == 1);
}

TEST(WorldTest, HierarchyInvalidNlevels) {
    World w({.nx = 16, .ny = 16, .seed = 42, .enable_spectral = true});
    EXPECT_THROW(w.hierarchy_init(1), InvalidArgument);
    EXPECT_THROW(w.hierarchy_init(9), InvalidArgument);
}
#endif

TEST(ConfigTest, Defaults) {
    Config cfg;
    EXPECT_EQ(cfg.nx, 64);
    EXPECT_EQ(cfg.ny, 64);
    EXPECT_EQ(cfg.nz, 1);
    EXPECT_FALSE(cfg.seamless);
}

TEST(PatternTest, Defaults) {
    Pattern p;
    EXPECT_EQ(p.weight, 1.0f);
}

TEST(PatternTest, Builder) {
    Pattern p(5, 2, 0.5f);
    EXPECT_EQ(p.id, 5);
    EXPECT_EQ(p.class_id, 2);
    EXPECT_EQ(p.weight, 0.5f);
}

TEST(DirTest, Opposite) {
    EXPECT_EQ(opposite(Dir::N), Dir::S);
    EXPECT_EQ(opposite(Dir::S), Dir::N);
    EXPECT_EQ(opposite(Dir::E), Dir::W);
    EXPECT_EQ(opposite(Dir::W), Dir::E);
    EXPECT_EQ(opposite(Dir::U), Dir::D);
    EXPECT_EQ(opposite(Dir::D), Dir::U);
}

TEST(WorldTest, SymmetricRuleAndPatternDef) {
    World w({.nx = 4, .ny = 4, .seed = 1});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    w.add_symmetric_rule(0, 1, Dir::N);
    ASSERT_NE(w.get_pattern_def(0), nullptr);
    EXPECT_EQ(w.get_pattern_def(0)->id, 0);
}

TEST(WorldTest, MoveLeavesUninitialized) {
    World w1({.nx = 2, .ny = 2, .seed = 1});
    World w2(std::move(w1));
    EXPECT_THROW((void)w1.nx(), Uninitialized);
}

TEST(WorldTest, ContradictionThrows) {
    World w({.nx = 2, .ny = 2, .seed = 1});
    w.add_pattern(Pattern(0, 0));
    EXPECT_THROW(w.collapse(), Contradiction);
}

TEST(WorldTest, ThreeD) {
    World w({.nx = 2, .ny = 2, .nz = 2, .seed = 2});
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W, Dir::U, Dir::D})
        w.add_rule(0, 0, d);
    w.collapse();
    EXPECT_TRUE(w.is_3d());
    EXPECT_EQ(w.export_grid().size(), 8u);
}

#ifdef WFC_USE_FAF
TEST(WorldTest, GumbelField) {
    Config cfg{.nx = 8, .ny = 8, .seed = 3, .enable_spectral = true};
    World w(cfg);
    w.add_pattern(Pattern(0, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W})
        w.add_rule(0, 0, d);
    w.set_gumbel_field(0.5, 9);
    w.collapse();
}

TEST(WorldTest, SpectralRequiresFlag) {
    World w({.nx = 4, .ny = 4, .seed = 1});
    EXPECT_THROW(w.set_matern_field(1.0, 1.0, 1), Uninitialized);
}
#endif

TEST(WorldTest, LearnFromSample) {
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

TEST(WorldTest, LargeCollapse) {
    World w({.nx = 32, .ny = 32, .seed = 42});
    w.add_pattern(Pattern(0, 0));
    w.add_pattern(Pattern(1, 0));
    for (auto d : {Dir::N, Dir::E, Dir::S, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(0, 1, d);
        w.add_rule(1, 0, d);
        w.add_rule(1, 1, d);
    }
    w.collapse();
    auto grid = w.export_grid();
    EXPECT_EQ(grid.size(), 1024);
}
