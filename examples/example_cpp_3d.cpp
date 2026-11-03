/* example_cpp_3d.cpp — 3D RAII collapse with U/D rules */
#include <entropic/wfcpp.hpp>
#include <iostream>

using namespace wfcpp;

int main() {
    std::cout << "=== WFC C++ 3D Example ===\n\n";

    try {
        World world(Config{.nx = 6, .ny = 6, .nz = 4, .seed = 123});

        world.add_pattern(Pattern(0, 0, 2.0f)); /* stone */
        world.add_pattern(Pattern(1, 1, 1.0f)); /* air */

        for (Dir d : {Dir::N, Dir::S, Dir::E, Dir::W, Dir::U, Dir::D}) {
            world.add_rule(0, 0, d);
            world.add_rule(1, 1, d);
            world.add_symmetric_rule(0, 1, d);
        }

        std::cout << "Collapsing " << world.nx() << "x" << world.ny()
                  << "x" << world.nz() << " (is_3d=" << world.is_3d() << ")...\n";
        world.collapse();

        auto grid = world.export_grid();
        for (uint32_t z = 0; z < world.nz(); z++) {
            std::cout << "\nLayer z=" << z << ":\n";
            for (uint32_t y = 0; y < world.ny(); y++) {
                for (uint32_t x = 0; x < world.nx(); x++) {
                    uint16_t p = grid[(z * world.ny() + y) * world.nx() + x];
                    std::cout << (p == 0 ? '#' : '.');
                }
                std::cout << '\n';
            }
        }

        int stone = 0, air = 0;
        for (auto p : grid) {
            if (p == 0) stone++;
            else if (p == 1) air++;
        }
        std::cout << "\nStone: " << stone << "  Air: " << air
                  << "  cells: " << world.num_cells() << "\n";
        std::cout << "\nDone!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
