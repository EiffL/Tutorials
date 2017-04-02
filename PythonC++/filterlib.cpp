#include <boost/python.hpp>
#include "filter.hpp"

using namespace boost::python;

// Defines a python module which will be named "mylib"
BOOST_PYTHON_MODULE(filter)
{
	  np::initialize();

    enum_<filter_type>("filter_type")
        .value("SOBEL_X", SOBEL_X)
        .value("SOBEL_Y", SOBEL_X)
        .value("LAPLACIAN", LAPLACIAN);
        
	  def("filter", &filter);
}
