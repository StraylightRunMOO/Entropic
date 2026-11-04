/* entropic_py.cpp — nanobind Python bindings for Entropic WFC engine */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <entropic/wfcpp.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace wfcpp;

NB_MODULE(_entropic, m) {
    m.doc() = "Entropic — Wave Function Collapse HPC Engine";

    /* Dir enum */
    nb::enum_<Dir>(m, "Dir")
        .value("N", Dir::N)
        .value("S", Dir::S)
        .value("E", Dir::E)
        .value("W", Dir::W)
        .value("U", Dir::U)
        .value("D", Dir::D);

    m.def("opposite", &opposite, "Return the opposite direction");

    /* Pattern */
    nb::class_<Pattern>(m, "Pattern")
        .def(nb::init<>())
        .def(nb::init<uint16_t, uint16_t, float>(),
             "id"_a, "class_id"_a = 0, "weight"_a = 1.0f)
        .def_rw("id", &Pattern::id)
        .def_rw("class_id", &Pattern::class_id)
        .def_rw("weight", &Pattern::weight)
        .def_rw("freq", &Pattern::freq)
        .def_rw("num_layers", &Pattern::num_layers)
        .def("__repr__", [](const Pattern& p) {
            return "Pattern(id=" + std::to_string(p.id) +
                   ", class_id=" + std::to_string(p.class_id) +
                   ", weight=" + std::to_string(p.weight) + ")";
        });

    /* Config */
    nb::class_<Config>(m, "Config")
        .def(nb::init<>())
        .def("__init__", [](Config* c, uint32_t nx, uint32_t ny, uint32_t nz,
                            uint32_t seed, bool seamless, bool enable_backtracking,
                            float spectral_weight, float copula_weight) {
            new (c) Config();
            c->nx = nx;
            c->ny = ny;
            c->nz = nz;
            c->seed = seed;
            c->seamless = seamless;
            c->enable_backtracking = enable_backtracking;
            c->spectral_weight = spectral_weight;
            c->copula_weight = copula_weight;
        }, "nx"_a = 64, "ny"_a = 64, "nz"_a = 1, "seed"_a = 0,
           "seamless"_a = false, "enable_backtracking"_a = false,
           "spectral_weight"_a = 1.0f, "copula_weight"_a = 1.0f)
        .def_rw("nx", &Config::nx)
        .def_rw("ny", &Config::ny)
        .def_rw("nz", &Config::nz)
        .def_rw("seed", &Config::seed)
        .def_rw("seamless", &Config::seamless)
        .def_rw("enable_backtracking", &Config::enable_backtracking)
        .def_rw("enable_spectral", &Config::enable_spectral)
        .def_rw("enable_copula", &Config::enable_copula)
        .def_rw("spectral_weight", &Config::spectral_weight)
        .def_rw("copula_weight", &Config::copula_weight)
        .def("__repr__", [](const Config& c) {
            return "Config(nx=" + std::to_string(c.nx) +
                   ", ny=" + std::to_string(c.ny) +
                   ", nz=" + std::to_string(c.nz) +
                   ", seed=" + std::to_string(c.seed) + ")";
        });

    /* CellView */
    nb::class_<CellView>(m, "CellView")
        .def("collapsed", &CellView::collapsed)
        .def("pattern_id", &CellView::pattern_id)
        .def("entropy", &CellView::entropy)
        .def("index", &CellView::index)
        .def("__repr__", [](const CellView& cv) {
            if (cv.collapsed())
                return "CellView(pattern=" + std::to_string(cv.pattern_id()) + ")";
            return std::string("CellView(uncollapsed)");
        });

    /* World */
    nb::class_<World>(m, "World")
        .def(nb::init<const Config&>(), "config"_a)
        .def("nx", &World::nx)
        .def("ny", &World::ny)
        .def("nz", &World::nz)
        .def("num_cells", &World::num_cells)
        .def("num_patterns", &World::num_patterns)
        .def("is_3d", &World::is_3d)
        .def("add_pattern", &World::add_pattern, "pattern"_a)
        .def("add_rule", &World::add_rule, "pat_a"_a, "pat_b"_a, "dir"_a)
        .def("add_symmetric_rule", &World::add_symmetric_rule,
             "pat_a"_a, "pat_b"_a, "dir"_a)
        .def("set_class_limit", &World::set_class_limit,
             "class_id"_a, "min_count"_a, "max_count"_a)
        .def("fix_cell", &World::fix_cell, "x"_a, "y"_a, "z"_a, "pat"_a)
        .def("learn_from_sample", [](World& w, nb::ndarray<uint16_t, nb::ndim<1>> arr,
                                     uint32_t sx, uint32_t sy, uint32_t sz) {
            std::vector<uint16_t> v(arr.data(), arr.data() + arr.shape(0));
            w.learn_from_sample(v, sx, sy, sz);
        }, "sample"_a, "sx"_a, "sy"_a, "sz"_a = 1)
        .def("learn_from_sample", [](World& w, std::vector<uint16_t> sample,
                                     uint32_t sx, uint32_t sy, uint32_t sz) {
            w.learn_from_sample(sample, sx, sy, sz);
        }, "sample"_a, "sx"_a, "sy"_a, "sz"_a = 1)
#ifdef WFC_USE_FAF
        .def("set_gumbel_field", &World::set_gumbel_field,
             "H"_a, "noise_seed"_a)
        .def("set_matern_field", &World::set_matern_field,
             "kappa"_a, "nu"_a, "noise_seed"_a)
#endif
#ifdef WFC_ENABLE_HIERARCHY
        .def("hierarchy_init", &World::hierarchy_init, "nlevels"_a)
        .def("hierarchy_set_wavelet", &World::hierarchy_set_wavelet,
             "family"_a, "dwt_levels"_a)
        .def("hierarchy_set_target", [](World& w, nb::ndarray<float, nb::c_contig> arr) {
            size_t n = 1;
            for (size_t i = 0; i < arr.ndim(); ++i) n *= arr.shape(i);
            w.hierarchy_set_target(arr.data(), static_cast<uint32_t>(n));
        }, "target"_a)
        .def("hierarchy_set_mode", &World::hierarchy_set_mode, "mode"_a)
        .def("hierarchy_adopt_spectral", &World::hierarchy_adopt_spectral)
        .def("hierarchy_residual_energy", &World::hierarchy_residual_energy, "level"_a)
        .def("hierarchy_collapse", &World::hierarchy_collapse,
             nb::call_guard<nb::gil_scoped_release>())
#endif
#ifdef WFC_ENABLE_SUSPENDERS
        .def("parallel_init", &World::parallel_init, "num_workers"_a = 0)
        .def("parallel_shutdown", &World::parallel_shutdown)
#endif
        .def("set_contradiction_callback", [](World& w, nb::handle cb) {
            if (cb.is_none()) {
                w.set_contradiction_callback(nullptr);
            } else {
                nb::object obj = nb::borrow<nb::object>(cb);
                w.set_contradiction_callback(
                    [obj](uint32_t x, uint32_t y, uint32_t z) {
                        nb::gil_scoped_acquire guard;
                        obj(x, y, z);
                    }
                );
            }
        }, "callback"_a.none())
        .def("collapse", &World::collapse,
             nb::call_guard<nb::gil_scoped_release>())
        .def("step_init", &World::step_init)
        .def("step", &World::step)
        .def("step_done", &World::step_done)
        .def("get_pattern", [](const World& w, uint32_t x, uint32_t y, uint32_t z) {
            return w.get_pattern(x, y, z);
        }, "x"_a, "y"_a, "z"_a = 0)
        .def("cell", &World::cell, "index"_a)
        .def("cell_at", &World::cell_at, "x"_a, "y"_a, "z"_a = 0)
        .def("export_grid", [](const World& w) {
            auto grid = w.export_grid();
            size_t nz = w.nz(), ny = w.ny(), nx = w.nx();
            uint16_t* data = new uint16_t[grid.size()];
            std::copy(grid.begin(), grid.end(), data);
            nb::capsule owner(data, [](void* p) noexcept {
                delete[] static_cast<uint16_t*>(p);
            });
            if (nz > 1) {
                size_t shape[3] = {nz, ny, nx};
                return nb::ndarray<nb::numpy, uint16_t, nb::ndim<3>>(
                    data, 3, shape, owner);
            } else {
                size_t shape[3] = {1, ny, nx};
                return nb::ndarray<nb::numpy, uint16_t, nb::ndim<3>>(
                    data, 3, shape, owner);
            }
        })
        .def("reset", &World::reset)
        .def("__repr__", [](const World& w) {
            return "World(" + std::to_string(w.nx()) + "x" +
                   std::to_string(w.ny()) + "x" +
                   std::to_string(w.nz()) + ", " +
                   std::to_string(w.num_patterns()) + " patterns)";
        });
}
