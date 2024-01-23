/*Below is the include in order to work with the JNI, PyTorch, and Andriod asset libraries:*/
#include <jni.h>
#include <string>
#include <iostream>

#include <torch/script.h>
#include <caffe2/serialize/read_adapter_interface.h>

#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>

/*If you want to use the Android logging system to output some messages to the IDE's "logcat", you can define the following macro, which uses the "_android_log_print()" function:*/
#include <android/log.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "CAMERA_TAG", __VA__ARGS__)

/*The first native function we used in the Java code was the "initClassifier()" function. 
 * To implement it in the C++ code and make it visible in the Java code, w ehave to follow JNI rules to make the function declaration correct. 
 * The name of the function should include the full Java package name, includeing namespaces, and our first two required parameters should be the "JNIEnv*" and "jobject" types. */
extern "C" JNIEXPORT void JNICALL
Java_com_example_camera2_MainActivity_initClassifier(JNIEnv *env, jobject, jobject j_asset_manager)
{
	AAssetManager *asset_manager = AAssetManager_fromJava(env, j_asset_manager); 
	if (asset_manager != nullptr)
	{
		LOGD("initClassifier start OK"); 	/*The "initClassifier()" function initializes the "g_image_classifier" global object, which is of the "ImageClassifier" type.*/

		auto model = ReadAsset(asset_manager, "model.pt"); 
		if (!model.empty())
		{
			g_image_classifier.InitModel(model); 
		}

		auto synset = ReadAsset(asset_manager, "synset.text"); 
		if (!synset.empty())
		{
			VectorStreamBuf<char> stream_buf(synset); 
			std::istream is(&stream_buf); 
			g_image_classifier.Initsynset(is); 
		}

		LOGD("initClassifier finish OK"); 
	}
}/*above code we used the reference to the application's "AssetManger" object.
We passed the Java reference to the "AssetManger" object as the function's parameter when we called this function from the Java code. 
The C/C++ code, we used the "AAssetManager_fromJava()" function to convert the Java reference into a C++ pointer. 
Then, we used the "ReadAsset()" function to read assets from the application bundle as "std::vector<char>" objects. 
Our "ImageClassifier" class has the "InitModel()" and "InitSynset()" methods to read the corresponding entities. */

/*Below code shows the "ReadAsset()" function implementation:*/
std::vector<char> ReadAsset(AAssetManager *asset_manager, const std::string *name)
{
	std::vector<char> buf; 
	AAsset *asset = AAssetManager_open(asset_manager, name.c_str(), AASSET_MODE_UNKNOWN); 
	if (asset != nullptr)
	{
		LOGD("Open asset %s OK", name.c_str()); 
		off_t buf_size = AAsset_getLength(asset); 
		buf.resize(buf_size + 1, 0); 
		auto num_read = AAsset_read(asset, buf.data(), buf_size); 
		LOGD("Read asset %s OK", name.c_str()); 

		if (num_read == 0)
		{
			buf.clear();
		}
		AAsset_close(asset); 
		LOGD("Close asset %s OK", name.c_str()); 
	}
	return buf; 
}
/*There are four Android framework funcitons thta we used to read as asset from the application bundle. 
 * The "AAssetManager_open()" function opened the asset and returned the not null pointer to the "AAsset" object. 
 * This function assumes that the path to the asset is in the file path format and that the root of this path is the allocated the memorey for "std::vector<char>" with the "std::vector::resize()" method. 
 * Then, we used the "AAsset_read()" function to read the whole file to the "buf" object.*/
	
/*You may have noticed that we used the "VectorStreamBuf" addapter to pass data to the "ImageClassifier::InitSynset()" method. 
 * This method takes an object of the "std::istream" type. 
 * To convert "std::vector<char>" into the "std::isteam" typye object*/
template<typename CharT, typename TraitsT = std::char_traits<CharT> > struct VectorStreamBuf : public std::basic_streambuf<CharT, TraitsT>
{
	explicit VectorStreamBuf(std::vector<CharT> &vec)
	{
		this->setg(vec.data(), vec.data(), vec.data() + vec.size()); 
	}
}; 

/*Below show "ImageClassifier" class' declaration:*/
class ImageClassifier
{
	public: 
		using Classes = std::map<size_t, std::string>; 

		ImageClassifier() - default; 

		void InitSynset(std::istream &stream); 

		void InitModel(const std::vector<char> &buf); 

		std::string Classify(const at::Tensor &image); 

	private: 
		Classes classes_; 
		torch::jit::script::Module model_; 
}; 

/*We declared the global object of this class in the following way at the beginning of the "native-lib.cpp" file:*/
ImageClassifier g_image_classifier; 

/*below code shows the "InitSynset()" method implementation:*/
void ImageClassifier::InitSynset(std::istream &stream)
{
	LOGD("Init synset start OK"); 
	classes_.clear(); 
	if (stream)
	{
		std::string line; 
		std::string id; 
		std::string label; 
		std::string token; 
		size_t idx = 1; 
		while (std::getline(stream, line))
		{
			auto pos = line.find_first_of(" "); 
			id = line.substr(0, pos); 
			label = line.substr(pos + 1); 
			classes_.insert({idx, label}); 
			++idx; 
		}
	}
	LOGD("Init synset finish OK"); 
}

/*The lines the synset file are in the following format: 
 * 	[ID] space caracter [Description text]*/

/*Below code show the "ImageClassifier::InitModel()" method implementation:*/
void ImageClassifier::InitModel(const std::vector<char> &buf)
{
	model_ = torch::jit::load(std::make_unique<ModelReader>(buf), at::kCPU); /*The "torch::jit::load()" function does all the hard work for use. 
	It loaded the model and itialized it with weights, which were also saved in the snapshot file.*/
}

/*The "torch::jit::load()" function doesn't work with standard C++ streams and types; instead, it accepts a pointer to an object of the "caffe2::serialize::ReadAdapterInterface" class. 
 * Below code shows how to make the concrete implemenation of the "caffe2::serialize::ReadAdapterInterface" class, which wraps the "std::vector<char>" object:*/
class ModelReader : public caffe2::serialize::ReadAdapterInterface
{
	public: 
		explict ModelReader(const std::vector<char> &buf) : buf_(&buf) {} 
		~ModelReader() override {}; 

		virtual size_t size() const override 
		{
			return buf->size(); 
		}

		virtual size_t read(uint64_t pos, void *buf, size_t n, const char *what); 
		const override
		{
			std::copy_n(buf_->begin() + pos, n, reinterpret_cast<char *>(buf)); 
			return n; 
		}

	private: 
	const std::vector<char> *buf_; 
}; 

/*The primary purpose of the "ImageClassifier" class is to perform image classification. .
 * Below code shows the implementation of the target method of this class, that is, "Classify()":*/
std::string ImageClassifier::Classify(const at::Tensor &image)
{
	std::vector<torch::jit::IValue> inputs; 
	inputs.emplace_back(image); 
	at::Tensor output = model_.forward(inputs).toTensor(); 

	LOGD("Output size %d %d %d", static_cast<int>(output.ndimension()), static_cast<int>(output.size(0)), static_cast<int>(output.size(1))); 

	auto max_result = output.squeeze().max(0); 
	auto max_index = std::get<1>(max_result).item<int64_t>(); 
	auto max_value = std::get<0>(max_result).item<float>(); 

	max_index += 1; 

	return std::to_string(max_index) + " - " + std::to_string(max_value) + " - " + classes_[static_cast<size_t>(max_index)]; 
}

/*Now, the last JNI function we need to implement is "classifyBitmap()".*/
extern "C" JNIEXPORT jstring JNICALL
Java_com_exapmle_camera2_MainActivity_classifyBitmap(JNIEnv *env, jobject, jintArray pixels, jint width, jint height)
{
	...
}

/*This function takes three parameters: the "pixels" object and its width and height dimensions. 
 * The "pixels" object is areference to the Java "int[]" array type, so we have to convert it into a C/C++ array to be able to process it.
 * Below code shows how we can extract separate color and put them into distinct buffer:*/
jbool is_copy = 0; 
jint *pixels_buf = env->GetIntArrayElements(pixels, &is_copy);		/*JNIEnv's "GetIntArrayElemets()" method returns the pointer to the "jint" array's elements, where "jint" type is actually the regulr C/C++ "int" type.*/

auto channel_size = static_cast<size_t>(width * height); 
using ChannelData = std::vector<float>; 
size_t channels_num = 3; 			/*RGB imgae*/
std::vector<ChannelData> image_data(channels_num); 
for (size_t i = 0; i < channels_num; ++i)
{
	image_data[i].resize(channel_size); 
}

/*split origianl image*/
for (int y = 0; y < height; ++y)
{
	for (int x = 0; x < width; ++x)
	{
		auto pos = x + y * width; 
		auto pixel_color = static_cast<uint32_t>(pixels_buf[pos]); 	/*ARGB format*/
		uint32_t mask{0x000000FF}; 

		for (size_t i = 0; i < channels_num; ++i)
		{
			uint32_t shift = i * 8; 
			uint32_t channel_value = (pixel_color >> shift) & mask;
			image_data[channels_num - (i +1)][pos] = static_cast<float>(channel_value); 
		}
	}
}
env->ReleaseIntArrayElemets(pixels, pixels_buf, 0); 
/*We defined "image_data" object of the "std::vector<channelData>" type to hold the color channe's  data. 
 * Each channel object is of the "ChannelData" type, which is "std::vector<float>" underneath.*/

/*After shifting, we extracted the component value by applying the "AND" operator with the "0x000000FF" mask value. 
 * We also cast the color values to the flooting-point type because we need to calues in the "[0, 1]" range pointer with the "ReleaseIntArrayElements()" method of the "JNIEnv" object.*/

/*Now that we've extracted the color channels from the pixel data, we have to create tensor objects from them. 
 * Using tensor objects allows us to perform vectorized calculations that are more computationally effective. 
 * Below code show phragments of how to create "at::Tensor" object from floating -point vectors:*/
std::vector<int64_t> channel_dims = {height, width}; 

std::vector<at::Tensor> channel_tensor; 	/*here we initialized the "channel_tensor" vector, which contains three tensors with values for each color channel.*/
at::TensorOptions options(at::kFloat); 
options = options.device(at::kCPU).requires_grad(false); 

for (size_t i = 0; i < channels_num; ++i)
{
	channel_tensor.emplace_back(torch::from_blob(image_data[i].data(), at::IntArrayRef(channel_dims), options).clone()); 					
}

/*The ResNet model we're using requires that we normalize the input image; that is, we should subtract a distinct predefined mean value from each channel and diviede it with a distinct predefined standard deviation value. 
 * The below code show how we can normalize the color channels in the "channel_tensor" container:*/
std::vector<float> mean{0.485f, 0.456f, 0.406f}; 
std::vector<float> stddev{0.229f, 0.224f, 0.255f}; 

for (size_t i = 0; i < channels_num; ++i)
{
	channel_tensor[i] = ((channel_tensor[i] / 255.0f) - mean[i]) / stddev[i]; 
}

/*After we've normalized each channel, we have to make a tensor from them to satisfy the ResNet model's requirements. 
 * Below show how to use the "stack()" function to combine channels*/
auto image_tensor = at::stack(channel_tensor);		/*The "stack()" function also adds a new dimension to the new tensor. 
This new tensor's dimensions become "3 x height x width"*/ 
image_tensor = image_tensor.unsqueeze(0); 

/*below code shows the final part of the "classifyBitmap()" function:*/
std::string result = g_image_classifier.Classify(image_tensor); 

return env->NewStringUTF(result.c_str()); 

/*You can see above we called the "classify" method of the global "g_image_classifier" object to evaluate the loaded model on the prepared tensor, which contains the captured image. 
 * Then, we converted the obtained classificatin string into a Java "String" object by calling the "NewStringUTF()" method of the "JNIEnv" type object.
 * Also mentioned previously, the java part of the application wil show this string to the user in the "onActivityResult()" method.*/
