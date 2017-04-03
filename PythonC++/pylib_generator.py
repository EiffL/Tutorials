#!/usr/bin/python
from pygccxml import parser
from pyplusplus import module_builder

# Configurations that you may have to change on your system
generator_path = "/usr/bin/castxml"
generator_name = "castxml"
compiler = "gnu"
compiler_path = "/usr/bin/gcc"

# Create configuration for CastXML
xml_generator_config = parser.xml_generator_configuration_t(
                                    xml_generator_path=generator_path,
                                    xml_generator=generator_name,
                                    compiler=compiler,
                                    compiler_path=compiler_path)

# List of all the C++ header of our library
header_collection = ["Bonjour.hpp"]

# Parses the source files and creates a module_builder object
builder = module_builder.module_builder_t(
                        header_collection,
                        xml_generator_path=generator_path,
                        xml_generator_config=xml_generator_config)

# Automatically detect properties and associated getters/setters
builder.classes().add_properties(exclude_accessors=True)

# Define a name for the module
builder.build_code_creator(module_name="pylib_auto")

# Writes the C++ interface file
builder.write_module('pylib_auto.cpp')
