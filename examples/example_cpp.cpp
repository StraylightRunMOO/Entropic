/* example_cpp.cpp — C++17 WFC example with RAII wrapper */
#include <entropic/wfcpp.hpp>
#include <iostream>
#include <vector>

using namespace wfcpp;

int main() {
    std::cout << "=== WFC C++17 Example ===\n\n";

    try {
        /* Create world */
        Config cfg{
            .nx = 16,
            .ny = 16,
            .nz = 1,
            .seed = 42,
            .seamless = false
        };
        World world(cfg);

        /* Define tiles */
        world.add_pattern(Pattern(0, 0, 5.0f));   /* grass */
        world.add_pattern(Pattern(1, 1, 3.0f));   /* water */
        world.add_pattern(Pattern(2, 2, 1.0f));   /* shore */

        /* Rules: grass<->grass, water<->water, shore connects to both */
        for (Dir d : {Dir::N, Dir::S, Dir::E, Dir::W}) {
            world.add_rule(0, 0, d);
            world.add_rule(1, 1, d);
            world.add_rule(0, 2, d);
            world.add_rule(2, 0, d);
            world.add_rule(1, 2, d);
            world.add_rule(2, 1, d);
            world.add_rule(2, 2, d);
        }

        /* Contradiction callback for debugging */
        world.set_contradiction_callback([](uint32_t x, uint32_t y, uint32_t z) {
            std::cout << "Contradiction at (" << x << ", " << y << ", " << z << ")\n";
        });

        /* Collapse */
        std::cout << "Collapsing 16x16 grid...\n";
        world.collapse();

        /* Export and print */
        auto grid = world.export_grid();
        std::cout << "\nResult:\n";
        std::cout << grid_to_string(world, grid);

        /* Stats */
        int grass = 0, water = 0, shore = 0;
        for (auto p : grid) {
            if (p == 0) grass++;
            else if (p == 1) water++;
            else if (p == 2) shore++;
        }
        std::cout << "\nCounts: grass=" << grass
                  << " water=" << water << " shore=" << shore << "\n";

        /* Demonstrate reset */
        std::cout << "\nResetting and re-collapsing with different seed...\n";
        world.reset();
        world.collapse();
        auto grid2 = world.export_grid();
        std::cout << grid_to_string(world, grid2);

        std::cout << "\nDone!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
