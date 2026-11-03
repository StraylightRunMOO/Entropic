/* example_cpp_advanced.cpp — Advanced C++ wrapper usage:
 * step-by-step collapse, learn_from_sample, CellView inspection,
 * class limits, contradiction callback, reset, and grid export. */
#include <entropic/wfcpp.hpp>
#include <iostream>
#include <iomanip>

using namespace wfcpp;

enum Tiles { FLOOR = 0, WALL = 1, DOOR = 2, PILLAR = 3 };

static const char GLYPH[] = {'.', '#', '+', 'O'};

static void print_grid(const World& w, const std::vector<uint16_t>& grid) {
    for (uint32_t y = 0; y < w.ny(); y++) {
        for (uint32_t x = 0; x < w.nx(); x++)
            std::cout << GLYPH[grid[y * w.nx() + x]];
        std::cout << '\n';
    }
}

int main() {
    constexpr uint32_t NX = 16, NY = 16;

    World w(Config{.nx = NX, .ny = NY, .nz = 1, .seed = 99});

    w.add_pattern(Pattern(FLOOR,  0, 5.0f));
    w.add_pattern(Pattern(WALL,   1, 3.0f));
    w.add_pattern(Pattern(DOOR,   1, 0.5f));
    w.add_pattern(Pattern(PILLAR, 2, 0.3f));

    /* --- Learn adjacency rules from a hand-crafted room --- */
    std::vector<uint16_t> sample = {
        1,1,2,1,1,1,1,1,
        1,0,0,0,0,0,0,1,
        1,0,3,0,0,3,0,1,
        1,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,1,
        1,0,3,0,0,3,0,1,
        1,0,0,0,0,0,0,1,
        1,1,1,1,2,1,1,1,
    };
    w.learn_from_sample(sample, 8, 8, 1);
    std::cout << "Learned rules from 8x8 room sample\n";

    /* --- Cap pillars to at most 12 --- */
    w.set_class_limit(2, 0, 12);
    std::cout << "Pillar class capped at 12 tiles\n";

    /* --- Contradiction callback --- */
    int contradictions = 0;
    w.set_contradiction_callback([&](uint32_t x, uint32_t y, uint32_t z) {
        (void)z;
        contradictions++;
        std::cout << "  contradiction at (" << x << "," << y << ")\n";
    });

    /* --- Step-by-step collapse --- */
    std::cout << "\n=== Step-by-step collapse ===\n";
    w.step_init();
    int steps = 0;
    while (!w.step_done()) {
        w.step();
        steps++;
    }
    std::cout << "Collapsed in " << steps << " steps\n";

    auto grid = w.export_grid();
    print_grid(w, grid);

    /* --- CellView inspection --- */
    std::cout << "\n=== CellView sample ===\n";
    for (uint32_t i = 0; i < 5; i++) {
        auto cv = w.cell(i);
        auto [cx, cy, cz] = idx_to_coords(w, i);
        std::cout << "  cell " << i << " (" << cx << "," << cy << "): "
                  << (cv.collapsed() ? GLYPH[cv.pattern_id()] : '?')
                  << " entropy=" << std::fixed << std::setprecision(2)
                  << cv.entropy() << "\n";
    }

    /* --- Count pillar tiles --- */
    int pillar_count = 0;
    for (auto t : grid)
        if (t == PILLAR) pillar_count++;
    std::cout << "\nPillar tiles: " << pillar_count << " (limit: 12)\n";

    /* --- Reset and re-collapse with full collapse() --- */
    std::cout << "\n=== Reset + full collapse ===\n";
    w.reset();
    contradictions = 0;
    try {
        w.collapse();
        auto grid2 = w.export_grid();
        print_grid(w, grid2);
        std::cout << "Second collapse succeeded\n";
    } catch (const Contradiction& e) {
        std::cout << "Second collapse contradicted: " << e.what() << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
