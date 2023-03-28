#include <vector>
#include <string>
#ifdef __ROOTCLING__
#pragma link C++ class std::vector<float>+;
#pragma link C++ class std::vector<std::string>+;
#pragma link C++ class std::vector<std::vector<bool> >+;
#pragma link C++ class std::vector<std::vector<int> >+;
#pragma link C++ class std::vector<std::vector<float> >+;
#pragma link C++ class std::vector<std::vector<double> >+;
#endif

