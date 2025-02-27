#include <torch/extension.h>
#include "gaussian_mixture.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<gmm::GMMVariant>(m, "GMMVariant")
        .value("GEM", gmm::GMMVariant::GEM)
        .value("VBEM", gmm::GMMVariant::VBEM)
        .value("CEM", gmm::GMMVariant::CEM);

    m.def("gem_core", &gmm::gem_core,
          py::arg("x"),
          py::arg("K"),
          py::arg("alpha"),
          py::arg("tol"),
          py::arg("max_iter"));
    
    m.def("vbem_core", &gmm::vbem_core,
          py::arg("x"),
          py::arg("K"),
          py::arg("alpha"),
          py::arg("tol"),
          py::arg("max_iter"));
    
    m.def("cem_core", &gmm::cem_core,
          py::arg("x"),
          py::arg("K"),
          py::arg("alpha"),
          py::arg("tol"),
          py::arg("max_iter"));
    
    m.def("hierarchical_gmm", 
          &gmm::gmm_hierarchical_implementation,
          py::arg("x"),
          py::arg("hierarchy_k"),
          py::arg("alpha"),
          py::arg("tol"),
          py::arg("max_iter"),
          py::arg("variant") = gmm::GMMVariant::GEM);
} 