#include <boost/python.hpp>
#include "filter.hpp"

using namespace boost::python;

// Defines a python module which will be named "mylib"
BOOST_PYTHON_MODULE(filter)
{
	  np::initialize();

	  def("filter", &filter);
}
