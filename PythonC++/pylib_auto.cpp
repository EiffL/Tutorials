// This file has been generated by Py++.

#include "boost/python.hpp"

#include "Bonjour.hpp"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(pylib_auto){
    { //::Bonjour
        typedef bp::class_< Bonjour > Bonjour_exposer_t;
        Bonjour_exposer_t Bonjour_exposer = Bonjour_exposer_t( "Bonjour", bp::init< std::string >(( bp::arg("msg") )) );
        bp::scope Bonjour_scope( Bonjour_exposer );
        bp::implicitly_convertible< std::string, Bonjour >();
        { //::Bonjour::greet
        
            typedef void ( ::Bonjour::*greet_function_type)(  ) ;
            
            Bonjour_exposer.def( 
                "greet"
                , greet_function_type( &::Bonjour::greet ) );
        
        }
        { //property "msg"[fget=::Bonjour::get_msg, fset=::Bonjour::set_msg]
        
            typedef ::std::string ( ::Bonjour::*fget)(  ) const;
            typedef void ( ::Bonjour::*fset)( ::std::string ) ;
            
            Bonjour_exposer.add_property( 
                "msg"
                , fget( &::Bonjour::get_msg )
                , fset( &::Bonjour::set_msg )
                , "get\\set property, built on top of \"std::string Bonjour::get_msg() const [member function]\" and \"void Bonjour::set_msg(std::string msg) [member function]\"" );
        
        }
    }
}