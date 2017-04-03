#include <iostream>
#include <string>

class Bonjour
{
    // Private attribute
    std::string m_msg;
public:
    // Constructor
    Bonjour(std::string msg):m_msg(msg) { }

    // Methods
    void greet() { std::cout << m_msg << std::endl; }

    // Getter/Setter functions for the attribute
    void set_msg(std::string msg) { this->m_msg = msg; }
    std::string get_msg() const { return m_msg; }
};
