/* example_cpp_wavelets.cpp — Spectral, hierarchy, and parallel C++ APIs.
 * Compiled only when ENTROPIC_USE_FAF=ON. Hierarchy / suspenders blocks
 * compile in when those CMake options are also on. */
#include <entropic/wfcpp.hpp>
#include <iostream>
#include <vector>

using namespace wfcpp;

static void add_two_tiles(World& w)
{
    w.add_pattern(Pattern(0, 0, 1.0f));
    w.add_pattern(Pattern(1, 1, 1.0f));
    for (Dir d : {Dir::N, Dir::S, Dir::E, Dir::W}) {
        w.add_rule(0, 0, d);
        w.add_rule(1, 1, d);
        w.add_symmetric_rule(0, 1, d);
    }
}

static void print_preview(const World& w, uint32_t stride)
{
    auto grid = w.export_grid();
    for (uint32_t y = 0; y < w.ny(); y += stride) {
        for (uint32_t x = 0; x < w.nx(); x += stride) {
            uint16_t p = grid[y * w.nx() + x];
            std::cout << (p == 0 ? '#' : '.');
        }
        std::cout << '\n';
    }
}

int main() {
    std::cout << "=== WFC C++ Wavelets Example ===\n\n";

    try {
#ifdef WFC_USE_FAF
        {
            std::cout << "--- Gumbel fBm then Matérn ---\n";
            World w(Config{
                .nx = 16, .ny = 16, .nz = 1,
                .seed = 42,
                .enable_spectral = true,
                .spectral_weight = 0.6f
            });
            add_two_tiles(w);
            w.set_gumbel_field(0.7, 12345);
            w.collapse();
            std::cout << "fBm:\n";
            print_preview(w, 1);

            w.reset();
            w.set_matern_field(1.0, 1.5, 99);
            w.collapse();
            std::cout << "\nMatérn:\n";
            print_preview(w, 1);
        }
#endif

#ifdef WFC_ENABLE_HIERARCHY
        {
            std::cout << "\n--- 3-level hierarchy (CDF 9/7) ---\n";
            World w(Config{
                .nx = 32, .ny = 32, .nz = 1,
                .seed = 7,
                .enable_spectral = true,
                .spectral_weight = 1.2f
            });
            add_two_tiles(w);
            w.hierarchy_init(3);
            w.hierarchy_set_wavelet(3, 0); /* FAF_WAVELET_CDF97, full depth */
            w.set_gumbel_field(0.7, 11);
            w.hierarchy_adopt_spectral(); /* one split of that fBm */
            w.hierarchy_collapse();
            std::cout << "coarsest residual energy "
                      << w.hierarchy_residual_energy(0) << "\n";
            print_preview(w, 2);
        }
#endif

#ifdef WFC_ENABLE_SUSPENDERS
        {
            std::cout << "\n--- Parallel DWT (64x64, 4 workers) ---\n";
            World w(Config{
                .nx = 64, .ny = 64, .nz = 1,
                .seed = 3,
                .enable_spectral = true,
                .spectral_weight = 0.8f
            });
            add_two_tiles(w);
            w.parallel_init(4);
            w.set_gumbel_field(0.65, 5);
            w.collapse();
            print_preview(w, 4);
            w.parallel_shutdown();
        }
#endif

        std::cout << "\nDone!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
