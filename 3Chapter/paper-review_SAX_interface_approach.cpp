struct Paper
{
	uint32_t id{0}; 
	std::string preliminary_decision; 
	std::vector<Review> reviews; 
}; 

using Papers = std::vector<Paper>; 

struct Review
{
	std::string confidence; 
	std::string evaluation; 
	uint32_t id{0}; 
	std::string language; 
	std::string orientation; 
	std::string remarks; 
	std::string text; 
	std::string timespan; 
}; 

/*Next, we declare a type for the object, which will be used by the parser to handle parsing events. 
 * This type should be inherited from the "rapidjson::BasedReaderHandler" base class, and we need to override virtual handler functions that the parser will call when a particular parsing event occurs*/
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>
...
struct ReviewsHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, ReviewsHandler>
{
	ReviewsHandler(Papers* papers) : papers_(papers) {}
	bool Uint(unsigned u); 
	bool String(const char* str, rapidjson::SizeType length, bool); /*copy*/
	bool Key(const char* str, rapidjson::SizeType length, bool); 	/*copy*/
	bool StartObject(); 
	bool EndObject(rapidjson::SizeType); 		/*memberCount*/
	bool StartArray(); 
	bool EndArray(rapidjson::SizeType); 		/*elementcount*/

	Paper paper_; 
	Review review_; 
	std::string key_; 
	Papers* papers_{nullptr}; 
	HandlerState state_{HandlerState::None}; 
}; 

/*Next, we can create the "rapidjson::FileReaderStream" object and initialize it with a handler to the opened file and with a buffer object that the parser will use for intermediate storage. 
 
* We use the "rapidjson::FileReadStream" object as the argument to the "Parse()" method of the "rapidjson::Reader" type object. 
 * The second is the object of the type we derived from "rapidjson::BaseReaderHandler"*/
auto file = std::unique_ptr<FILE, void(*)(FILE*)>(fopen(filename.c_str(), "r"), [](FILE* f)
		{
			if (f)
			::fclose(f); 
		}); 
if (file)
{
	char readBuffer[65536]; 
	rapidjson::FileReadStream is(file.get(), readBuffer, sizeof(readBuffer)); 
	rapidjson::Reader reader; 
	Papers papers; 
	ReviewsHandler handler(&papers); 
	auto res = reader.Parse(is, handler); 
	if (!res)
	{
		throw std::runtime_error(rapidjson::GetParseError_En(res.Code())); 
	}
	return papers; 
} else 
{
	throw std::invalid_argument("File can't be opened " + filename); 
}

/*The event handler works as a state machine. 
 * In one state, we populate it with the "Review" objects, and in another one with the "Papers" objects, and there are states for other events*/
enum class HandlerState
{
	None, 
	Global,
	PaperArray, 
	Paper, 
	ReviewArray, 
	Review
}; 

/*We parse the unsigned "unit" values only for the "Id" attributes of the "Paper" and the "Review" objects, and we update these values according to the current state and the previously parsed key*/
bool Uint(unsigned u)
{
	bool res{true}; 
	try
	{
		if (state == HandlerState::Paper && key_ == "id")
		{
			paper_.id = u; 
		} else if (state_ == HandlerState::Paper && key_ == "id")
		{
			review_.id = u; 
		} else 
		{
			res = false; 
		}
	} catch (...)
	{
		res = false; 
	}
	key_.clear(); 
	return res; 
}

/*String values also exist in both types of objects, so we do the same check to update corresponding values*/
bool String(const char* str, rapidjson::SizeType length, bool)
{
	bool res{true}; 
	try
	{
		if (state_ == HandlerState::Paper && key_ == "preliminary_decision")
		{
			paper.preliminary_decision = std::string(str, length); 
		} else if (state_ == HandlerState::Review && key_ == "confidence")
		{
			review_.confidence = std::string(str, length);
		} else if (state_ == HandlerState::Review && key_ == "evaluation")
		{
			review_.evaluation = std::string(str, length);
		} else if (state_ == HandlerState::Review && key_ == "lan")
		{
			review_.language = std::string(str, length); 
		} else if (state_ == HandlerState::Review && key_ == "orientation")
		{
			review_.orientation = std::string(str, length); 
		} else if (state_ == HandlerState::Review && key_ == "remarks")
		{
			review_.remarks = std::string(str, length); 
		} else if (state_ == HandlerState::Review && key_ == "text")
		{
			review_.text = std::string(str, length); 
		} else if (state_ == HandlerState::Review && key_ == "timespan")
		{
			review_.timespan = std::string(str, length); 
		} else 
		{
			res = false ;
		}
		key_.clear(); 
		return res; 
}

/*The event handler for the JSON "key" attruibute stores the "key" value to the appropriate variable, which we use to identify a current object in the parsing process*/
bool Key(const char* str, rapidjson::SizeType length, bool)
{
	key_ = std::string(str, length); 
	return true; 
}

/*We base the current implementation on the knowledge of the structure of the current JSON file: there is no array of "Paper" objects, and each "Paper" object includes an array of review. 
 * It is one of the limitations of the SAX interface we need to know the structure of the document to implement all event handlers currectly.*/
bool StartObject()
{
	if (state_ == HandlerState::None && key_.empty())
	{
		state_ = HandlerState::Global; 
	} else if (state_ = HandlerState::PapersArray && key_.empty())
	{
		state_ = HandlerState::Paper; 
	} else if (state_ == HandlerState::ReviewArray && key_.empty())
	{
		state_ = HandlerState::Review; 
	} else 
	{
		return false; 
	}
	return true; 
}

/*The "EndObject" event handler, populate arrays of "Paper" and "Review" objects according to the current stats. 
 * Also we switch the current state back to the previous one(its a stack)*/
bool EndObject(rapidjson::SizeType)
{
	if (state_ == HandlerState::Global)
	{
		state_ = HandlerState::None; 
	} else if (state_ == HandlerState::Paper)
	{
		state_ = HandlerState::PaperArray; 
		papers_->push_back(paper_); 
		paper_ = Paper(); 
	} else if (state_ == HandlerState::Review)
	{
		state_ = HandlerState::ReviewArray; 
		paper_.reviews.push_back(review); 
	} else 
	{
		return false; 
	}
	return true; 
}

/*IN the "StartArray" event handler, we switch the current state to a new one according to the current state value*/
bool StartArray()
{
	if (state_ == HandlerState::Global && key_ == "paper")
	{
		state_ = HandlerState::PaperArray; 
		key_.clear(); 
	} else if (state_ == HandlerState::Paper && key_ == "review")
	{
		state_ = HandlerState::ReviewArray; 
		key_.clear();
	} else 
	{
		return false; 
	}
	return true; 
}

/*The "EndArray" event handler, we switch the current state to the previous one based on out knowledge of the document structure*/
bool EndArray(rapidjson::SizeType)
{
	if (state_ == HandlerState::ReviewArray)
	{
		state_ = HandlerState::Paper; 
	} else if (state_ == HandlerState::PapersArray)
	{
		state_ = HandlerState::Global; 
	} else 
	{
		return false; 
	}
	return true; 
}


