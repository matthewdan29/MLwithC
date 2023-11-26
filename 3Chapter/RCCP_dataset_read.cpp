#include <csv.h>

/*Now we define an object of the type "io::CSVReader".
 * We must define the number of columns as a  template parameter.*/
const uint32_t colums_num = 5; 
io::CSVReader<columns_num> csv_reader(file_path); 

/*Next, we define containers for storing the values we read*/
std::vector<std::string> categorical_column; 
std::vector<double> values; 

/*Then, to make our code more generic and gather all information about column types in one place, we introduce the following helper types and functions. 
 * We define a tuple object that describes values for a row*/
using RowType = std::tuple<double, double, double, double, std::string>; 
RowType row; 

/*The "read_row()" method takes a variable number of parameters of different types. 
 * We automatic parameter filling by using the "std::index_sequence" type with the "std::get" function, */

template <std::size_t... Idx, typename T, typename R>
bool read_row_help(std::index_sequence<Idx...>, T& row, R& r)
{
	return r.read_row(std::get<Idx>(row)...); 
}

/*The second helper function uses a similar technique for transforming a row tuple object to our value vectors*/
template<std::size_t... Idx, typename T>
void fill_values(std::index_sequence<Idx...>, T& row, std::vector<double>& data)
{
	data.insert(data.end(), {std::get<Idx>(row)...}); 
}

/*Now, we define a loop where we continuouly read values and move them to our containter. 
 * Next, we check the return value of the "read_row()" method, which tells us if the read was successful or not. 
 * A "false" return value means that we have reached the end of the file*/
try
{
	bool done = false; 
	while (!done)
	{
		done = !read_row_help(std::make_index_sequence<std::tuple_size<RowType>::vale>{}, row, csv_reader); 
		if (!done)
		{
			categorical_column.push_back(std::get<4>(row)); 
			fill_values(std::make_index_sequence<colums_num - 1>{}, row, values); 
		}
	}
} catch (const io::error::no_digit& err)
{
	/*ignore badly formatted smaples*/
	std::cerr << err.what() << std::endl; 
}

/*Notice that we moved only four values to our vector of doubles because the last column contains string objects that we put to another vector of categorical values. */
