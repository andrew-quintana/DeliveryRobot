//
// Created by Andrew Quintana on 11/30/23.
//
#include <pybind11/pybind11.h>
#include "AprilTagSensor.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(AprilTagSensor, m) {
    m.doc() = "optional module docstring";

    py::class_<AprilTagSensor>(m, "Sensor")
    .def(py::init<double, double, int>())
    .def("run", &AprilTagSensor::run, py::call_guard<py::gil_scoped_release>())
    .def_readonly("v_data", &AprilTagSensor::v_data, byref)
    .def_readonly("v_gamma", &AprilTagSensor::v_gamma, byref)
    ;
}