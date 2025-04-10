#include <torch/extension.h>
#include "qem_partitioning.h"

namespace py = pybind11;

PYBIND11_MODULE(qem_cpp, m) {
    m.doc() = "Python binding for QEM partitioning algorithm";
    
    m.def("qem_partitioning", &qem::qem_partitioning, 
          py::arg("points"), 
          py::arg("neighbors"), 
          py::arg("distances"), 
          py::arg("normals"), 
          py::arg("k_init"), 
          py::arg("reg"),
          py::arg("qem_tol"),
          py::arg("max_iterations") = 5,
          "Compute QEM for point cloud partitioning");
} 