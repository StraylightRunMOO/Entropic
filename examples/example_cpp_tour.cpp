/* example_cpp_tour.cpp — Remaining C++ wrapper surface:
 * symmetric rules, seamless torus, copula via c_ptr(), backtracking,
 * pattern metadata, CellView, move, grid_to_string / idx_to_coords. */
#include <entropic/wfcpp.hpp>
#include <iostream>
#include <utility>

using namespace wfcpp;

static void add_full_sym(World& w, uint16_t npat)
{
    for (Dir d : {Dir::N, Dir::E}) {
        for (uint16_t a = 0; a < npat; a++)
            for (uint16_t b = 0; b < npat; b++)
                w.add_symmetric_rule(a, b, d);
    }
}

int main() {
    std::cout << "=== WFC C++ API Tour ===\n\n";

    try {
        /* --- Move a freshly constructed world --- */
        World src(Config{
            .nx = 16, .ny = 16, .nz = 1,
            .seed = 99,
            .enable_backtracking = true,
            .enable_copula = true,
            .seamless = true,
            .copula_weight = 1.0f
        });
        World w = std::move(src);
        std::cout << "Moved world: " << w.nx() << "x" << w.ny()
                  << " seamless=" << w.seamless()
                  << " patterns-to-add...\n";

        w.add_pattern(Pattern(0, 0, 1.0f));
        w.add_pattern(Pattern(1, 1, 1.0f));
        add_full_sym(w, 2);

        const wfc_pattern_def_t* def = w.get_pattern_def(0);
        std::cout << "get_pattern_def(0): id=" << def->id
                  << " class=" << def->class_id
                  << " weight=" << def->weight << "\n";
        std::cout << "opposite(Dir::N)=" << static_cast<int>(opposite(Dir::N))
                  << "  num_classes=" << w.num_classes() << "\n";

#ifdef WFC_ENABLE_COPULA
        /* Copula rho is not wrapped; use the C escape hatch. */
        wfc_world_t* raw = w.c_ptr();
        raw->copula_marginal[0] = 0.5f;
        raw->copula_marginal[1] = 0.5f;
        raw->copula_z[0] = wfc_inv_cdf_normal(0.5f);
        raw->copula_z[1] = wfc_inv_cdf_normal(0.5f);
        for (int d = 0; d < 4; d++) {
            raw->copula_rho[d][0][0] = 0.8f;
            raw->copula_rho[d][1][1] = 0.8f;
            raw->copula_rho[d][0][1] = -0.4f;
            raw->copula_rho[d][1][0] = -0.4f;
        }
        std::cout << "copula_rho set via c_ptr()\n";
#endif

        w.set_contradiction_callback([](uint32_t x, uint32_t y, uint32_t z) {
            std::cout << "  contradiction at (" << x << "," << y << "," << z << ")\n";
        });

        std::cout << "\nCollapsing torus with copula + backtrack...\n";
        w.collapse();

        auto grid = w.export_grid();
        std::cout << grid_to_string(w, grid);

        std::cout << "\nCellView + idx_to_coords:\n";
        for (uint32_t i = 0; i < 4; i++) {
            auto cv = w.cell(i);
            GridCoords c = idx_to_coords(w, i);
            uint32_t back = coords_to_idx(w, c.x, c.y, c.z);
            char glyph = cv.collapsed()
                ? static_cast<char>('0' + cv.pattern_id())
                : '?';
            std::cout << "  idx " << i << " -> (" << c.x << "," << c.y << ")"
                      << " glyph=" << glyph
                      << " entropy=" << cv.entropy()
                      << " roundtrip=" << back << "\n";
        }
        std::cout << "get_pattern(0,0)=" << w.get_pattern(0, 0)
                  << "  rng_state=" << w.rng_state() << "\n";

        std::cout << "\nReset + re-collapse...\n";
        w.reset();
        w.collapse();
        std::cout << grid_to_string(w, w.export_grid());

        std::cout << "\nDone!\n";
    } catch (const Contradiction& e) {
        auto [x, y, z] = e.where();
        std::cerr << e.what() << " at (" << x << "," << y << "," << z << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
