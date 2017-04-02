#include <iostream>
#include <string>

class Bonjour
{
    // Private attribute
    std::string msg;

public:
    
    // Constructor
    Bonjour(std::string msg):msg(msg) { }
    
    // Methods
    void greet() { std::cout << msg << std::endl; }
    
    // Getter/Setter functions for the attribute
    void set(std::string msg) { this->msg = msg; }
    std::string get() { return msg; }
};
