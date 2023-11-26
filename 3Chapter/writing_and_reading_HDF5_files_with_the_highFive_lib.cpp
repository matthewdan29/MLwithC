#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

/*Now, we have to create a file object where we will write our dataset*/
HighFive::File file(file_name, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate); 

/*After we have a file object we can start creating groups. 
 * We define a group of papers that should hold all paper objects*/
auto paper_group = file.createGroup("papers"); 

/*Next, we iterate through an array of papers and create a group for each paper object with two attributes: 
 * 		1) "Id" number attribute
 * 		2) "perliminary_decision" attribute of the string type*/

for (const auto& paper : papers)
{
	auto paper_group = papers_group.createGroup("paper_" = std::to_string(paper.id)); 
	std::vector<uint32_t> id = {paper.id}; 
	auto id_attr = paper_group.createAttribute<uint32_t>("id", HighFive::DataSpace::From(id)); 

	id_attr.write(id); 
	auto dec_attr = paper_group.createAttribute<std::string>("preliminary_decision", HighFive::DataSpace::From(paper.preliminary_decision)); 
	dec_attr.write(paper.preliminary_decision); 
	
	/*After creating the attribute, we have to put in its value with the "write()" method. 
	 * Notice that the "HighFive::DataSpace::From" function automatically detcects the size of the attribute value. 
	 * The size is the amount of memory required to hold the attributes value. 
	 * For each "paper_group" we create a corresponding group of reviews*/
	auto reviews_group = paper_group.createGroup("reviews"); 

	/*We insert into each "reviews_group" a detaset of numerical values of "confidence", "evaluation", and "orientation" fields. 
	 * For the dataset we define the "DataSpace" of size 3 and define a storage type as a 32 bit integer*/

	std::vector<size_t> dims = {3}; 
	std::vector<int32_t> values(3); 
	for (const auto& r : paper.reviews)
	{
		auto dataset = reviews_group.createDataSet<int32_t>( std::to_string(r.id), HighFive::DataSpace(dims)); 
		values[0] = std::stoi(r.confidence); 
		values[1] = std::stoi(r.evaluation); 
		values[2] = std::stoi(r.orientation); 
		dataset.write(values); 
	}
}


/*Having the file in the HDF5 format, we can consider the "HighFive" library interface for file reading. 
 * As the first step, we again create the "HighFive::File" object but with attributes for reading*/
HighFive::File file(file_name, HighFive::File::ReadOnly); 

/*Next, we use the "getGroup()" method to get the top level "papers_group" object*/
auto papers_group = file.getGroup("papers"); 

/*We need to get a list of all nested objects in this group because we can access objects only by their names. */
auto papers_names = papers_group.listObjectNames(); 

/*Using a loop we iterate over all "papers_group" objects in the "papers_group" container*/
for (const auto& pname : papers_names)
{
	auto paper_group = papers_group.getGroup(pname); 
	...
}

/*For each() "paper" object we read its attributes, and the memory space required for the attribute value. 
 * Because each attribute can be multidimensional, we should take care of it and allocate an approriate container, */
std::vector<uint32_t> id; 
paper_group.getAttribute("id").read(id); 
std::cout << id[0]; 

std::string decision; 
paper_group.getAttribute("preliminary_decision").read(decision); 
std::cout << " " << decision << std::endl; 

/*For reading datasets, we can use the same approach: get the "reviews" group then get a list of dataset names, and finally, read each dataset in a loop */
auto reviews_group = paper_group.getGroup("reviews"); 
auto reviews_names = reviews_group.listObjectNames(); 
std::vector<int32_t> values(2); 
for (const auto& rname : reviews_names)
{
	std::cout << "\t review: " << rname << std::endl; 
	auto dataset = reviews_group.getDataSet(rname); 
	auto selection = dataset.select( {1}, {2}); /*or use just dataset.read method to get whole data*/
	selection.read(values); 
	std::cout << "\t\t evaluation: " << values[0] << std::endl; 
	std::cout << "\t\t orientation: " << values[1] << std::endl; 
}

/*Using these techniques, we can read and transform any HDF5 datasts. 
 * This file format allows us to work ony with part of the required data and not to load the whole file to the memeory. 
 * Also becuse this is a binary format, its reading is more efficient than reading large text files.*/
