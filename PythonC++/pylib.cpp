#include <boost/python.hpp>
#include "Bonjour.hpp"

using namespace boost::python;

// Defines a python module which will be named "pylib"
BOOST_PYTHON_MODULE(pylib)
{
    // Declares class Bonjour, specifying the constructor, the greet function
    // and declaring msg as an attribute visible from python, which can be queried
		// and set through the C++ get and set functions.
	class_< Bonjour >("Bonjour", init<std::string>())
	  .def("greet", &Bonjour::greet)
	  .add_property("msg", &Bonjour::get_msg, &Bonjour::set_msg);
}
