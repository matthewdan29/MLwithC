/*We read the labels file with the "ReadLabels" function*/

/*This function opens the file in binary mode and reads the header records, the magic number, and the numeber of items in the file. 
 * It also reads all the items directly to the C++ vector. 
 * The most important part is to correctly read the header records.*/
void MNISTDataset::ReadLabels(const std::string& labels_file_name)
{
	std::ifstream labels_file(labels_file_name, std::ios::binary | std::ios::badbit); 
	if (labels_file)
	{
		uint32_t magic_num = 0; 
		uint32_t num_items = 0; 
		if (read_header(&magic_num, labels_file) && read_header(&num_items, labels_file))
		{
			labels_.resize(static_cast<size_t>(num_items)); 
			labels_file.read(reinterpret_cast<char*>(labels_.data()), num_items); 
		}
	}
}


/*This function reads the values from the input stream in our case, the file stream and flips the endianness. 
 * This function also assumes that header records are 32 bit integer values. 
 * In a different scenario, we woul dhave to think of other way to flip the endianness.*/
template <class T> bool read_header(T* out, std::istream& stream)
{
	auto size = static_cast<std::streamsize>(sizeof(T)); 
	T value; 
	if (!stream.read(reinterpret_cast<char*>(&value), size))
	{
		return false; 
	} else 
	{
		/*flip endianness*/
		*out = (value << 24) | ((value << 8) & 0x00FF000) | ((value >> 8) & 0x0000FF00) | (value >> 24); 
		return true; 
	}
}
