#include <boost/python.hpp>
#include "Bonjour.hpp"

using namespace boost::python;

// Defines a python module which will be named "mylib"
BOOST_PYTHON_MODULE(mylib)
{
    // Declares class Bonjour, specifying the constructor, the greet function
    // and declaring msg as an attribute visible from python, which can be set/get
    // with the get and set functions.
	class_< Bonjour >("Bonjour", init<std::string>())
	  .def("greet", &Bonjour::greet)
	  .add_property("msg", &Bonjour::get, &Bonjour::set);
}
