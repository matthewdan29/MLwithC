/*First, we have to develop parser and datat loader classes to move the dataset to memory in a format suitble for use with PyTorch*/

/* namespace fs = std::filesystem; */

/*Below is the interface for the reader class:*/
class ImdbReader 
{
	Public: 
	ImdbReader(const std::string& root_path); 

	size_t GetPosSize() const; 
	size_t GetNegSize() const; 
	size_t GetMaxSize() const; 

	using Review = std::vector<std::string>; 
	const Review& GetPos(size_t index) const; 
	const Review& GetNeg(size_t index) const; 

	private: 
	using Reviews = std::vector<Review>; 

	void ReadDirectory(const std::string& path, Reviews& reviews); 

	private: 
	Reviews pos_samples_; 
	Reviews neg_samples_; 
	size_t max_size_{0}; 
}; 

/*we will assume that the object of this class should be initialized with the path to the root folder where one of the datasets is placed (the training set or test set). 
 * We can can initialize this like below*/
int main(int argc, char** argv)
{
	if (argc > 0)
	{
		auto root_path = fs::path(argv[1]); 
		...
		ImdbReader train_reader(root_path / "train");
		ImdbReader test_reader(root_path / "test"); 
	}
	...
}

/*The most important parts of this clas are the constructor and the "ReadDirectory" methods.
 * The constructor is the main point wherein we fill the containers, "pos_samples_" and "neg_samples_", with actual reviews from the "pos" and "neg" folders:*/
namespace fs = std::filesystem; 
...
ImdbReader::ImdbReader(const std::string& root_path)
{
	auto root = fs::path(root_path); 
	auto neg_path = root / "neg"; 
	auto pos_path = root / "pos"; 
	if (fs::exists(neg_path) && fs::exists(pos_path))
	{
		auto neg = std::async(std::launch::async, [&](){ReadDirectory(neg_path, neg_samples_); });
		auto pos = std::async(std::launch::async, [&](){ ReadDirectory(pos_path, pos_samples_); }); 
		neg.get(); 
		pos.get(); 
	} else 
	{
		throw std::invalid_argument("ImdbReader incorrect path"); 
	}
}

/*The "ReadDirectory" method implements the logic for iterating files in the given directory. 
 * It also reads them, tokenizes lines, and fills the dataset container, which is then passed as a function parameter. 
 * Below show "ReadDirectory" method's implementation*/
void ImdbReader::ReadDirectory(const std::string& path, Review& reviews)
{
	std::regex re("[^a-zA-Z0-9]"); 
	std::sregex_token_iterator end; 

	for (auto& entry : fs::directory_iterator(path))
	{
		if (entry.is_regular_file())
		{
			std::ifstrem file(entry.path()); 
			if (file)
			{
				std::string text; 
				{
					std::stringstream buffer; 
					buffer << file.rdbuf(); 
					text = buffer.str(); 
				}

				std::sregex_token_iterator token(text.begin(), text.end(), re, -1); 
				Reviews words; 
				for (; token != end; ++token)
				{
					if (token->length() > 1)
						/*dont use one letter words, 'I' and 'a' might be a problem if the network wasn't connected horizontal*/
						words.push_back(*token); 
				}
			}

			max_size_ = std::max(max_size_, words.size()); /*this is making sure the vector is big to fit the biggest words length. 
			Remember we are padding smaller words with 0's */
			reviews.push_back(std::move(words));
		}
	}
}

/*After, we've read the train and test datasets, we need to build a word vocabulary where each string representing a word matches a unique index.
 * We are going to use such a vocabulary to convert string-based reviews into integerbased representations that can be used with linear algebra abstractions.
 * We reduce the amount of noise so we are using the most common words with the most frequently. method for calculating the frequencies of all words by placing them in a unordered hash map object:*/
using WordFrequencies = std::unordered_map<std::string, size_t>; 

/*We can calculate frequencies by accumulating the number of times words appear in the second number of the pair from the map. 
 * This is done by iterating through all of the words in the review: */
void GetWordFrequencies(const ImdbReader& reader, WordsFrequencies& frequencies)
{
	for (size_t i = 0; i < reader.GetPosSize(); ++i)
	{
		const ImdbReader::Review& review = reader.GetPos(i); 
		for (auto& word : review)
		{
			frequencies[word] += 1; 
		}
	}

	for (size_t i = 0; i < reader.GetNegSize(); ++i)
	{
		const ImdbReader::Review& review = reader.GetNeg(i); 
		for (auto& word : review)
		{
			frequencies[word] += 1; 
		}
	}
}

/*"GetwordsFrequencies()" is used in the following way:*/
WordsFrequencies words_frequencies; 
GetWordsFrequencies(train_reader, words_frequencies); 
GetWordsFrequencies(test_reader, words_frequencies); 

/*After we have calculated the number of occurrences of each word in the datasets, we can select a specifi number of the most frequently used ones.
 * Let's set the size of the vocabulary 25,000 words:*/
int64_t vocab_size = 25000; 

/*To sort by frequencies and keep it contained in the unorder hash map rep we use a "std::vector" class. 
 * Then, we can use the standard sorting algorithm with a custom comparison function. 
 * This concept if fully implemented in the "SelectTopFrequencies" function:*/
void SelectTopFrequencies(WordsFrequencies& vocab, int64_t new_size)
{
	using FreqItem = std::pair<size_t, WordsFrequencies::iterator>; 
	std::vector<FreqItem> freq_items; 
	freq_items.reserve(vocab.size()); 
	auto i = vocab.begin(); 
	auto e = vocab.end(); 
	for (; i != e; ++i)
	{
		freq_items.push_back({i->second, i}); 
	}
	
	std::sort( freq_items.begin(), freq_items.end(), [](const FreqItem& a, const FreqItem& b) {return a.first < b.first; }); 

	std::reverse(freq_items.begin(), freq_items.end()); 

	freq_items.resize(static_cast<size_t>(new_size)); 

	WordsFrequencies new_vocab; 

	for (auto& item : freq_items)
	{
		new_vocab.insert({item.second->first, item.first}); 
	}

	vocab = new_vocab;
}

/*The standard library's "sort" function assumes that a passed comparison function returns true if the first argument is less than the second. 
 * So, after sorting, we reversed the result in order to move the most frequent words to the beginning of the container. 
 * Then, we simply resized the "freq_items" container to desired length.
 * The last step of this function was creating the new "wordsFrequencies" type object from the items representing the most frquently used words. 
 * Also, we replaced the content of the original "vocab" object with the "new_vocab" object content. 
 * Below shows how to use this function: */
SelectTopFrequncies(words_frequencies, vocab_size); 

/*Before we assign indices to the words, we have to decide how we are going to generate emeddings for them. 
 * This is an important issue because the indices we asign will be used to access the word embeddings.*/

/*We need to create a parser for the downloaded embeddings. 
 * The downloaded embeddings file contains one key-value pair per line, where the key is the word and the value is the 100-dimensional vector. 
 * All the items in the line are seprated by spaces, so the format look likes "x0x1x2...x99"*/
/*Below defines the "class" interface for the GloVe embedding's parser:*/
class GloveDict 
{
	public:
		GloveDict(const std::string& file_name, int64_t vec_size); 
		torch::Tensor Get(const std::string& key) const; 
		torch::Tensor GetUnkown() const; 

	private: 
		torch::Tensor unkown_; 
		std::unordered_map<std::string, torch::Tensor> dict_; 
}; 


/*The "GloveDict" class constructor takes the filename and the size of the embedding vector. 
 * There are two methods being used here. 
 * The "Get" method, "GetUnkown", returns the tensor representing the embedding for the words that don't exist in the embeddings list. 
 * In our case, this is zero */
/*The main work is done by the constructor of the class, where we read a file with GloVe vectors, parse it, and initialize the "dict_" map object with words in the keys role and embed tensors as values: */
GloveDict::GloveDict(const std::string& file_name, int64_t vec_size)
{
	std::ifstrem file; 
	file.exceptions(std::ifstrem::badbit); 
	file.open(file_name); 
	if (file)
	{
		auto sizes = {static_cast<long>(vec_size)}; 
		std::string line; 
		std::vector<float> vec(static_cast<size_t>(vec_size));	/*"std::vector<float>" holds the embeddeding vector values*/ 
		unkown_ = torch::zeros(size, torch::dtype(torch::kFloat));	/*"unknown_" tensor for unknown words*/
		std::string key; 
		std::string token; 
		while (std::getline(file, line))
		{
			if (!line.empty())
			{
				std::stringstrem line_stream(line); 
				size_t num = 0; 
				while (std::getline(line_strem, token, ' '))
				{
					if (num == 0)
					{
						key = token; 
					} else 
					{
						vec[num - 1] = std::stof(token);
					}

					++num; 
				}

				assert(num == (static_cast<size_t>(vec_size) + 1)); 
				torch::Tensor tvec = torch::from_blob(vec.data(), sizes, torch::TensorOptions().dtype(torch::kFloat)); 
				dict_[key] = tvec.clone(); 
			}
		}
	}
}

/*Now, we have everything we need to create a vocabulary class that can associate a word with a unique index, and the index with a vector embedding.*/
class Vocabulary
{
	public: 
	Vocabulary(const WordsFrequencies& words_frequencies, const GloveDict& glove_dict); 

	int64_t GetIndex(const std::string& word) const;	/*"GetIndex" returns the index for the input word*/
	int64_t GetPaddingIndex() const;			/*"GetPaddingIndex" method returns the index of the embedding, which can be used for padding.*/
	torch::Tensor GetEmbeddings() const;			/*"GetEmbeddings" returns a tensor containing all embeddings (in rows) in the same order as the word indices.*/
	int64_t GetEmbeddingsCount() const; 			/*"GetEmbeddingCount" method returns the total count of the embeddings.*/

	private; 
	std::unordered_map<std::string, size_t> words_to_index_map_; 
	std::vector<torch::Tensor> embeddings_; 
	size_t unk_index_; 
	size_t pad_index_; 
}; 

/*Below shows how "Vocabulary" constructor is implemented:*/
Vocabulary::Vocabulary(const WordsFrequencies& words_frequencies, const GloveDict& glove_dict)
{
	words_to_index_map_reserve(words_frequencies.size()); 
	embeddings_.reserve(words_frequencies.size()); 

	unk_index_ = 0; 
	pad_index_ = unk_index_ + 1; 

	embeddings_.push_back(glove_dict.GetUnknown()); 	/*unknown*/
	embeddings_.push_back(glove_dict.GetUnknown()); 	/*padding*/

	size_t index = pad_index_ + 1; 
	for (auto& wf : words_frequencies)
	{
		auto embedding = glove_dict.Get(wf.first); 
		if (embedding.size(0) != 0)
		{
			embeddings_.push_back(embedding); 
			words_to_index_map_.insert({wf.first, index}); 
			++index; 
		} else 
		{
			words_to_index_map_.insert({wf.first, unk_index_}); 
		}
	}
}

/*In this method, we populated the "words_to_index_map_" and "embeddings_" containers.
 * Firs, we inserted two zero-valued tensors into the "embeddings_" container; one for the GloVe unknown word and another for padding values:*/
embeddings_.push_back(glove_dict.GetUnknown()); 	/*unknown*/
embeddings_.push_back(glove_dict.GetUnknown()); 	/*padding*/

/*Notice hwo the "index" value is initialized and incremented; it starts with 2 because the 0 index is occupied for unknown embedding and 1 index is occupied for padding value embedding:*/
unk_index_ = 0; 
pad_index_ = unk_index_ + 1; 
...
size_t index = pad_index_ + 1; 

/*Notice that we only increased the index after we inserted a new embedding tensor into the "embeddings_" object. 
 * In the opposite case, when an embedding for the word was not found, the word was associated with the unkonwn value index. 
 * The next important method in the "Vocabulary" class in the "GetEmbeddings" method, which value index. 
 * The next important method in the "Vocabulary" class in the "GetEmbeddings" method, which makes a single tensor from a vector of embedding tensors.*/
at::Tensor Vocabulary::GetEmbeddings() const 
{
	at::Tensor weights = torch::stack(embeddings_); 
	return weights; 
}

/*Now, our dataset class should return a pair of training tensors: one representing the encoded text and another containing its length. 
 * Also, we need to develop a custom function to convert the vector of tensors in a batch into one single tensor. 
 * This function is required if we want to make PyTorch compatible with custom training data.*/
/*Now, we define the "ImdbSample" type for custom training data sample. 
 * We will use this with the "torch::data::Dataset" type: */
using ImdbData = std::pair<torch::Tensor, torch::Tensor>; 
using ImdbSample = torch::data::Example<ImdbData, torch::Tensor>; 

/*"IdbData" represents the training data and has two tensors for test sequence and length. 
 * "ImdbSample" represents the whole sample with a taret value. 
 * A tensor contains a 1 or 0 for positive or negative sentiment, respectively. */

/*below shows the "ImdbDataset" class declaration:*/
class ImdbDataset : public torch::data::Dataset<ImdbDataset, ImdbSample>
{
	public: 
		ImdbDataset(ImdbReader* reader, Vocabulary* vocabulary, torch::DeviceType device);

		/*torch::data::Datset implementation*/
		ImdbSample get(size_t index) override; 
		torch::optional<size_t> size() const override; 

	private: 
		torch::DeviceType device_{torch::DeviceType::CPU}; 
		ImdbReader* reader_{nullptr}; 
		Vocabulary* vocabulary_{nullptr}; 
}; 

/*We inherited our dataset class form the "torch::data::Dataset" class so that we can use it for data loader initialization. 
 * The Objects of our "ImdbDataset" class should be initialized with the "ImdbReader" and "Vocabulary" class instances. 
 * We also added the "device" parameter of "torch::DeviceType" into the constructor to tell the object where to place the training object in CPU or GPU memory. 
 * In the constructor, we store the pointer to input objects and the device type. 
 * We overrode two methods from the "torch::data::Dataset" class: the "get" and "size" methods*/

/*below shows the implemention the "size" method:*/
torch::optional<size_t> ImdbDataset::size() const 
{
	return reader_->GetPosSize() + reader_->GetNegSize();
}

/*The "size" method returns the number of reviews in the "ImdbReader" object. 
 * The "get" method has a  more complicated implementation than the previous one*/
ImdbSample ImdbDataset::get(size_t index)
{
	torch::Tensor target; 
	const ImdbReader::Review* review{nullptr}; 
	if (index < reader_->GetPosSize())
	{
		review = &reader_->GetPos(index); 
		target = torch::tensor(1.f, torch::dtype(torch::kFloat).device(device_).requires_grad(false)); 
	} else 
	{
		review = &reader_->GetNeg(index - reader_->GetPosSize()); 
		target = torch::tensor(0.f, torch::dtype(torch::kFloat).device(device_).requires_grad(fales)); 
	}
}

/*After we got the correct index, we also got the corresponding text review and assigned its address to the "review" pointer and initialized the "target" tensor. 
 * The "torch::tensor" function was used to initialize the target tensor. 
 * This function takes an arbitrary numeric value and tensor options such as a type and a device. 
 * Notice that we set the "requires_grad" option to false because we don't need to calculate the gradient for this variable. 
 * Below code shows the continuation of the "get" methods implementation: */
/*encode text*/
std::vector<int64_t> indices(reader_->GetMaxSize()); 
size_t i = 0; 

for (auto& w : (*review))
{
	indices[i] = vocabulary_->GetIndex(w); 
	++i; 
}

/*Here, we encoded the review text from string words to their indices. 
 * We defined the "indices" vector of integer values in order to store the encoding of the maximim possible length. 
 * Then, we filled it in the cycle by applying the "GetIndex" method of the vocabulary object to each of the words. 
 * Notice that we used the 'i' variable to count the number of words we encode.
 * The use of this variable was required because other positions in the sequence will be padded with a particular padding index. */
/*pad text to same size*/
for (; i < indices.size(); ++i)
{
	indices[i] = vocabulary_->GetPaddingIndex(); 
}

/*When we've initialized all the data we need for one training sample, we have to convert it into a "torch::Tensor" object. 
 * For this purpose, we can use already known functions, namely "torch::from_blob" and "torch::tensor". 
 * The "torch::from_blob" function takes the pointer for raw numeric data, the dimensions container, and tensor options. 
 * below shows how we use these functions to create the final tensor object at the end of the "get" method's implementation: */
auto data = torch::from_blob(indices.data(), {static_cast<int64_t>(reader_->GetMaxSize())}, torch::dtype(torch::kLong).requires_grad(false)); 

auto data_len = torch::tensor(static_cast<int64_t>(review->size()), torch::dtype(torch::kLong).requries_grad(false)); 

return {{data.clone().to(device_), data_len.clone()}, target.squeeze()}; 

/*Below shows how we initialize data loaders for the training and test dataset for the "get" method that was just demostrated*/
torch::DeviceType device = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU; 

...
/*create datasets*/
ImdbDataset train_dataset(&train_reader, &vocab, device); 
ImdbDataset test_dataset(&test_reader, &vocab, device); 

/*init data loader*/
size_t batch_size = 32; 
auto train_loader = torch::data::make_data_loader(train_dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)); 

auto test_loader = torch::data::make_data_loader(test_dataset, torch::data::DataLoaderOptions().batch_size(batch_size).worker(4)); 

/*Before, we move on, we need to define one more helper function, which converts the batch vector of tensors into one tensor. 
 * This conversion is needed to vectorize the calculation for better utilization of hardware resources, in order to improve performance. 
 * Notice that when we initialized the data loaders with the "make_data_loader" fucntion, we didn't use the mapping and transform methods for datasets objects as in the previous exapmle. 
 * This was done because, by default, PyTorch can't automatically transform arbitrary types (in our case, the "ImdbData pair type") into tensors. 
 * Below shows the begining of the "MakeBatchTensors" function's implementation: */

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MakeBatchTensors(const std::vector<ImdbSample>& batch)
{
	/*prepare batch data*/
	/*we are going to split the single vector "ImdbSample" object into three
	 * 	1) "text_data" - which contains all text
	 * 	2) "text_lengths" - which contains the corresponding lengths
	 * 	3) "label_data" - which contains the target value*/
	std::vector<torch::Tensor> text_data; 
	std::vector<torch::Tensor> text_lengths; 
	std::vector<torch::Tensor> label_data; 
	for (auto& item : batch)
	{
		text_data.push_back(item.data.first); 
		text_lengths.push_back(item.data.second); 
		label_data.push_back(item.target); 
	}

	/*Then, we need to sort them in decreasing order of text length. 
	 * This order is a requirement of the "pack_padded_sequence" function, which we will use in our model to transform padded sequences into packed ones to improve performance. 
	 * We can't simultaneously sort three containers in C++, so we have to use a custom approach based on a defined permutation. 
	 * Below show how to we applied this approach while continuing to implement the method*/
	std::vector<std::size_t> permutation(text_lengths.size()); 
	std::iota(permutation.begin(), permutation.end(), 0); 
	std::sort(permutation.begin(), permutation.end(), [&](std::size_t i, std::size_t j)
			{
				return text_lengths[i].item().toLong() < text_lengths[j].item().toLong(); 
			}); 
	std::reverse(permutation.begin(), permutation.end()); 

	/*Here, we defined the "permutation" vector of indices with a number of items equal to the batch size. 
	 * Then, we filled it consistently with numbers starting from 0, and sorted it with the standard "std::sort" algorithm function, but with a custom comparison functor, which compares the lengths of sequences with correspondent indices. 
	 * Notice that to get the raw value from the "torch::Tensor" type object, the "item()" and "toLong()" methods were used. 
	 * Also, because we needed to decreasing order of items, we used the "stdreverse" algorithm. 
	 * Below shows how we used the "permentation" object to sort three containers in the same way:*/
	auto appy_permutation = [&permutation](const std::vector<torch::Tensor>& vec)
	{
		std::vector<torch::Tensor> sorted_vec(vec.size()); 
		std::transform(permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i) {return vec[i]; }); 
		return sorted_vec; 
	}; 
	text_data = appy_permutation(text_data); 
	text_lengths = appy_permutation(text_lengths); 
	label_data = appy_permutation(label_data); 

	/*when all the batch vectors have been sorted in the required order, we can use the "torch::stack" function to concatenate each of them into the sinle tensor with an additional dimension. 
	 * Below shows hwo we used this fuction to create the final tensor object
	 * This is the final part of the "MakeBatchTensors" method's implementation:*/
	torch::Tensor texts = torch::stack(text_data); 
	torch::Tensor lengths = torch::stack(text_lengths); 
	torch::Tensor labels = torch::stack(label_data); 

	return {texts, lengths, labels}; 
}
/*At this point, we have written all the code required to parse and prepare the training data. 
* Now, we can ceate classes for our RNN model. 
* We are going to based our model on the LSTM architecture. 
* There is a module called "torch::nn::LSTM" in the PyTorch C++ API for this purpose. 
* The problem is that this module can't work with packed sequences. 
* There is a standalone function called "torch::lstm" that can do this, so we need to create our custom module to combine the "torch::nn::LSTM" module and the "torch::lstm" function so that we can work with packed sequences. 
* Such an approach causes our RNN to only process the non-padded elements of our sequences. 
* Below show the "PackedLSTMImpl" class' declaration and the "PackedLSTM" module's definition:*/
class PackedLSTMImpl : public torch::nn::Module
{
	public: 
	explicit PackedLSTMImpl(const torch::nn::LSTMOptions& options);

	std::vector<torch::Tensor> flat_weights() const; 

	torch::nn::RNNOutput forward(const torch::Tensor& input, const torch::Tensor& lengths, torch::Tensor state = {}); 

	const torch::nn::LSTMOptions& options() const; 

	private: 
	torch::nn::LSTM rnn_ = nullptr; 	
}; 
TORCH_MODULE(PackedLSTM); 

/*The "PackedLSTM" module definition uses the "PackedLSTMImpl" class as the module function's implementation. 
 * Also, notice that the "PackedLSTM" module definition differs from the "torch::nn::LSTM" module in that the "forward" function takes the additional parameter, "lengths". 
 * The implementation of this module is based on the code of the "torch::nn::LSTM" module form the "PyTorch" library. 
 * The "flat_weights" and "forward" function weere mostly copied from the "PyTorch" library's soruce code. 
 * We overrode the "flat_weights" function because it is hidden in the base class, and we can access it from the "torch::nn::LSTM" module. */

/*below shows the constructor's implementation of "PackedLSTMImpl"*/
PackedLSTMImpl::PackedLSTMImpl(const torch::nn::LSTMOptions& options)
{
	rnn_ = torch::nn::LSTM(options); 
	register_module("rnn", rnn_); 
}

/*Below shows the "flat_weights" method's implementation:*/
std::vector<torch::Tensor> PackedLSTMImpl::flat_weights() const
{
	std::vector<torch::Tensor> flat; 

	const auto num_directions = nn_->options.bidirectional_ ? 2 : 1; 
	for (int64_t layer = 0; layer < rnn_->options.layers_; layer++)
	{
		for (auto direction = 0; direction < num_directions; directions++)
		{
			const auto layer_idx = static_cast<size_t>((layer * num_directions) + direction); 
			flat.push_back(rnn_->w_ih[layer_idx]); 
			flat.push_back(rnn_->w_hh[layer_idx]); 
			if (rnn_->options.with_bias_)
			{
				flat.push_back(rnn_->b_ih[layer_idx]); 
				flat.push_back(rnn_->b_hh[layer_idx]); 
			}
		}	
	}
	return flat; 
}

/*The "forward" method is also a copy of the same method from the "torch::nn::LSTM" module, but it used a different overload of the "torch::lstm" function.
 * We can see that the main logic in the "forward" method is to initialize the cell state if it is not defined and called the "torch::lstm" function. 
 * Notice that all the methods in this class consider the "options.bidirectional" flag in order to configure the dimensions of the weights and state tensors. 
 * Also notice that the module's state is a combined tensor from two tensor: 
 * 	1) The hidden state 
 * 	2) The cell state */
/*Below shows the "forward" method's implementation:*/
torch::nn::RNNOutput PackedLSTMImpl::forward(const torch::Tensor& input, const at::Tensor& lengths, torch::Tensor state)
{
	if (!state.defined())
	{
		const auto max_batch_size = lengths[0].item().toLong(); 
		const auto num_directions = rnn_->options.bidirectional_ 2 : 1;
		state = torch::zeros({2, rnn_->options.layers_ * num_directions, max_batch_size, rnn_->options.hidden_size_}, input.options()); 
	}

	torch::Tensor output, hidden_state, cell_state; 
	std::tie(output, hidden_state, cell_state) = torch::lstm(input, lengths, {state[0], state[1]}, flat_weights(), rnn_->options.with_bias_, rnn_->options.layers_, rnn_->options.dropout_, rnn_->is_training(), rnn_->options.bidirectional_); 
	return {output, torch::stack({hidden_state, cell_state})}; 
}


/*Our RNN model cna be configured so that it's multilayer and bidirectional. 
 * These properties can significantly imporve model performance for the sentiment analysis task.*/
/*Below in the code notice we defineed the "embeddings_weights_" class member, which is of the "torch::autograd::Variable" type. 
 * This was done because we used the "torch::embedding" function to convert the input batch sequence's items into embeddings automatically. 
 * We can use the "torch::nn::Embedding" module for this purpose, but the C++ API can't use pre-traained values. 
 * This is why we used the "torch::embedding" function directly. 
 * We also used the "torch::autograd::Variable" type instead of a simple tensor because we need to calucate the gradient for our module during a training process.*/
/*Below shows how we can define our RNN model with the "SentimentRNN" class: */
class SentimentRNNImpl : public torch::nn::Module
{
	public: 
		SentimentRNNImpl(int64_t vocab_size, int64_t embedding_dim, int64_t hidden_dim, int64_t output_dim, int64_t n_layers, bool bidirectional, double dropout, int64_t pad_idx); 

		void SetPretrainedEmbeddings(const torch::Tensor& weights); 

		torch::Tensor forward(const torch::Tensor& text, const at::Tensor& length); 

	private: 
		int64_t pad_idx_{-1}; 
		torch::autograd::Variable embeddings_weights_; 
		PackedLSTM rnn_ = nullptr; 
		torch::nn::Dropout dropout_ = nullptr; 
}; 

TORCH_MODULE(SentimentRNN); 

/*Below the explaination is the "SentimentRNNImpl" class constructor's implementation: */
/*IN the constructor of our module, we initialized the based blocks of our network. 
 * We used the "register_parameter" method of the "torch::nn::Module" class to create the "embeddings_weights_" object, which is filled with the empty tensor. 
 * Registration makes automatically calculating the gradient possible. 
 * Notice that the one dimension of the "embeddings_weights_" object is equal to the vocabulary length, while the other one is equal to the ength of the embedding vector (100, in our case). 
 * The "rnn_" object is initialized with the "torch::nn::LSTMOptions" type object. 
 * we defined the length of the embedding, the number of hidden dimensions (number of hidden neurons in the LSTM module layers), the number of RNN layers, the flag that tells us whether the RNN is bidirectional or not, an dspecified the reularization parameter (the dropot factor value). */


/*The implementation of the "forward" method takes two tensors as input parameters. 
 * One is the text sequences, which are "[sequence length x batch size]" is size, while the other is text lengths, which are "[batch size x 1]" in size. 
 * First, we applied the "torch::embedding" function to our text sequences. 
 * This function converts indexed sequences into one with embedding values. 
 * It aslo takes "embeddings_weights_" as a parameter. 
 * "embeddings_weights_" is the tensor that contains our pre-trained embeddings.
 * The "pad_idx_" parameter tells us what index points to the padding value embedding. 
 * The result of calling this function is "[sequence length x batch size x 100]". 
 * We also applied the dropout module to the embedded sequences to perform regularization. */
/*Then, we converted the padded embedded sequences into packed ones with the "torch::_packed_padded_sequence" function. 
 * This function takes the padded sequences with their lenghts and returns a pair of new tensors with different sizes, which also represent packed sequences and packed lengths, correspondingly. 
 * We used packed sequences to improve the performance of the model. */

/*After, we passed the packed sequences and their lengths into the "PackedLSTM" module's forward functioin. 
 * THis module processed the input sequences with the RNN and returned an object of the "torch::nn::RNNOutput" type with two numbers: 
 * 	1) output
 * 	2) state
 * The "state" member is in the following format: "{hidden_state, cell_state}"*/
/*The hidden state has the following shape: "{num layers * num directions x batch size x hid dim}". 
 * The number of directions is 2 in the case of a bidirectional RNN. 
 * RNN layers are ordered as follows: 
 * "[forward_layer_0, backward_layer_0, forward_layer_1, backward_layer 1, ... forward_layer_n, backward_layer n]". */
torch::Tensor SentimentRNNImpl::forward(const at::Tensor& text, const at::Tensor& length)
{
	auto embedded = dropout_(torch::embedding(embeddings_weights_, text, pad_idx_)); 

	torch::Tensor packed_text, packed_length; 
	std::tie(packed_text, packed_length) = torch::_pack_padded_sequence(embedded, length.squeeze(1), false); 

	auto rnn_out = rnn_->forward(packed_text, packed_length); 

	auto hidden_state = rnn_out.state.narrow(0, 0, 1); 
	hidden_state.squeeze_(0); 	/*remove 0 dimension equals to 1 after narrowing*/
	/*take last hidden layers state*/
	auto last_index = rnn_->options().layers(). - 2; 
	hidden_state = at::cat({hidden_state.narrow(0, last_index, 1).squeeze(0), hidden_state.narrow(0, last_index + 1, 1).squeeze(0)}, 1); 

	auto hidden = dropout_(hidden_state); 

	return fc_(hidden); 
}

/*below the code shows how to get the hidden states for the last (top) layers*/
auto last_index = rnn_->options().layers() = 2; 
hidden_state = at::cat({hidden_state.narrow(0, last_index, 1).squeeze(0), hidden_state.narrow(0, last_index + 1, 1).squeeze(0)}, 1); 

/*The last step of the "forward" function was applying the dropout and passing the result to the fully connected layer*/
auto hidden = dropout_(hidden_state); 
return fc_(hidden); 

/*The following code shows how we can initialize the model:*/
int64_t hidden_dim = 256; 
int64_t output_dim = 1; 
int64_t n_layers = 2; 
bool bidirectional = true; 
double dropout = 0.5; 
int64_t pad_idx = vocab.GetPaddingIndex(); 

SentimentRNN model(vocab.GetEmbeddingsCount(), embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, pad_idx); 

/*Next, important step in the model configuration process is initializing the pre-train embeddings. 
 * Below shows how to use the "SetPretrainedEmbeddings" method to do so:*/
model->SetPretrainedEmbeddings(vocab.GetEmbeddings()); 

/*The "SetPretrainedEmbeddings" method is implemented in the following way below*/
void SentimentRNNImpl::SetPretrained embeddings(const at::Tensor& weights)
{
	torch::NoGradGuard guard; 
	embeddings_weights_.copy_(weights); 
}

/*When the model has been initialzed and configured, we can begin training. 
 * The necessary part of the training process is an optimizer object 
 * In this example, we will use the Adam optimization algorithm. 
 * The name Adam is derived from the adaptive moment estimation. 
 * This algorithm usually results in a better a faster convergence in comparison with pure stochastic gradient descent. 
 * below shows how to define an instance of the "torch::optim::Adam" class*/
double learning_rate = 0.01; 
torch::optim::Adam optimizer(model->paramters(), torch::optim::AdamOptions(learning_rate)); 

/*Now, we can move the model to a computational device such as a GPU: */
model->to(device); 

/*Then, we can start training the model. 
 * Training will be performed for 100 epochs over all the samples in the training dataset. */
int epochs = 100; 
for (int epoch = 0; epoch < epochs; ++epoch)
{
	TrainModel(epoch, model, optimizer, *train_loader); 
	TestModel(epoch, model, *test_loader); 
}

/*The "TrainModel" function will be implemented in a statndardized way for training neural networks with PyTorch. */
/*below is the code*/
void TrainModel(int epoch, SentimentRNN& model, torch::optim::Optimizer& optimizer, torch::data::StatelessDataLoader<ImdbDataset, torch::data::samplers::RandomSampler>& train_loader); 

/*Below shows how to enable training mode for the model:*/
model->train(); 		/*switch to the training mode*/

/*Below shows the beginning of the "TrainModel" function's implementation:*/
double epoch_loss = 0; 
double epoch_acc = 0; 
int batch_index = 0; 
for (auto& batch : train_loader)
{
	...
}

/*Below series of code shows the implementation of a training cycle's iteration:*/

/*	1) First, we clear the previous gradients from the optimizer:*/
optimizer.zero_grad(); 

/*	2) Then, we convert the batch data into distinct tensors: */
torch::Tensor text, lengths, labels; 
std::tie(texts, lengths, labels) = MakeBatchTensors(batch); 

/*	3) Now that we have the sample texts and lengths, we can perform the forward pass of the model:*/
torch::Tensor prediction = model->forward(texts.t(), lengths); 
prediction.squeeze_(1); 

/*	4) Now that we have the predictions from our model, we use the "squeeze_" function to remove any unnecessary dimensions so that the model's compatible with the loss function. 
 *	Notice that the "squeeze_" function has an underscore, which means that the function is evaluated in place, without any aditional memory begin allocated. */

/*	5) Then, we compute a loss value to estmate the error of our model:*/
torch::Tensor loss = torch::binary_cross_entropy_with_logits(prediction, labels, {}, {}, Reduction::Mean); 
	/*Here, we used the "torch::binary_cross_entropy_with_logits" function, which measures the binary cross-entropy between the "prediction" logits and the target "labels". 
	 * This function already includes a sigmoid calculation. 
	 * This is whyour model returns the output from the linear full conection layer. 
	 * We also specificed the reduction type in order to apply to the loss output. 
	 * Losses from each sample in the batch are summed and divided by the number of elements in the batch.*/

/*	6) Then, we compute the gradients for our model and update its parameters with these gradients:*/
loss.backward(); 
optimizer.step(); 

/*	7) One of the final steps of the training function is to accumulate the loss and accuracy values for averaging:*/
auto loss_value = static_cast<double>(loss.item<float>()); 
auto acc_value = static_cast<double>(BinaryAccuracy(prediction, labels)); 

epoch_loss += loss_value; 
epoch_acc += acc_value; 

/*Below we used the cusom "BinaryAccuracy" function for the accuracy calculation.*/
float BinaryAccuracy(const torch::Tensor& preds, const torch::Tensor& target)
{
	auto rounded_preds = torch::round(torch::sigmoid(preds)); 
	auto correct = torch::eq(round_preds, target).to(torch::dtype(torch::kFloat)); 
	auto acc = correct.sum() / correct.size(0); 
	return acc.item<float>(); 
}

/*In this function, we applied "torch::sigmoid" to the predictions of our model.
 * This operation converts the logits values into values we can interpret as a label (1 or 0), but because these values are floating points, we applied the "torch::round" function to them. 
 * The "torch::round" function rounds the input values to the closest integer.
 * Then, we compared the predicted labels with the target values using the "torch::eq" function. 
 * This operation gave us an initialized tenser, with 1 where labels matched and 0 otherwise. 
 * We calculated the ratio between the number of all labels in the batch and the number of correct predictions as an accuracy value.*/

/*Below shows the end of the training function's implementation:*/
std::cout << "Epoch: " << epoch << " | Loss: " << (epoch_los / (batch_index - 1)) << " | ACC: " << (epoch_acc / (batch_index - 1)) << std::endl; 

/*Below the "TestModel" function's implementation, which looks pretty similar to the "TrainModel" function: */
void TestModel(int epoch, SentimentRNN& model, torch::data::StatelessDataLoader<ImdbDataset, torch::data::samplers::RandomSampler>& test_loader)
{
	torch::NoGradGuard guard; 
	double epoch_loss = 0; 
	double epoch_acc = 0; 
	model->eval(); 				/*switch to the evaluation mode Iterate the data loader to get batches from the dataset*/
	int batch_index = 0; 
	for (auto& batch : test_loader)
	{
		/*prepared batch data*/
		torch::Tensor texts, lengths, labels; 
		std::tie(texts, lengths, labels) = MakeBatchTensors(batch); 

		/*Forward pass the model on the input data*/
		torch::Tensor prediction = model->forward(texts.t(), lengths); 
		prediction.squeeze_(1); 

		/*Compute a loss value to estimate error of our model*/
		torch::Tensor loss = torch::binary_cross_entropy_with_logits(prediction, labels, {}, {}, Reduction::Mean); 

		auto loss_value = static_cast<double>(loss.item<float>()); 
		auto acc_value = static_cast<double>(BinaryAccuracy(prediction, labels)); 

		epoch_loss += loss_value; 
		epoch_acc += acc_value; 

		++batch_index; 
	}

	std::cout << "Epoch: " << epoch << " | Test Loss: " << (epoch_loss / (batch_index - 1)) << " | Test ACC: " << (epoch_acc / (batch_index - 1)) << std::endl; 
}

/*The main differences regarding this function are that we used the "test_loader" object for data, switch the model the evaluation state with the "model->eval()" call, and we didn't use any optimization operations. 
 *
 * This RNN architecture, with the settings we used, results in 85% accuracy in the sentiment analysis of movie reviews.*/
