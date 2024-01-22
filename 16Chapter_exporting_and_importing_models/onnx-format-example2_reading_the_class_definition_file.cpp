/*Another important function that was used in the ONNX format example was a function that can reade class definitions from a synset file.
 * We used the "ReadClasses" function in this example to load the map of objects. 
 * The function is trivial and reads the synset file line by line. 
 * In such a file, each line contains a number and class description strig, separated with the space character.*/
using Classes = std::map<size_t, std::string>; 
Classes ReadClasses(const std::string& file_name)
{
	Classes classes; 
	std::ifstream file(file_name); 
	if (file)
	{
		std::string line; 
		std::string id; 
		std::string label; 
		std::string token; 
		size_t idx = 1; 
		while (std::getline(file, line))
		{
			std::stringstream line_stream(line); 
			size_t i = 0; 
			while (std::getline(line_stream, token, ' '))
			{
				switch (i)
				{
					case 0: 
						id = token; 
						break; 
					case 1: 
						label = token; 
						break; 
				}
				token.clear(); 
				++i; 
			}
			classes.insert({idx, label}); 
			++idx;
		}
	}
	return classes; 
}

/*Notice that we used the "std::getline" function in the internal "while" loop to tokenize a single line string. 
 * We did this by specifiying th third parameter that defines the delimiter charater value. */
