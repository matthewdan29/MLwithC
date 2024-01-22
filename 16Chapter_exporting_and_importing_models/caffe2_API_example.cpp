/*to be able to use the Caffe2 C++ API, we have to use the following headers*/
#include <caffe2/core/init.h>
#include <caffe2/onnx/backend.h>
#include <caffe2/utils/proto_utils.h>

/*We still need to link our program to the "libtorch.so" library*/
/*First, we need to initialize the Caffe2 library: */
caffe2::GlobalInit(&argc, &argv); 

/*Then, we need to load the Protobuf model representation. 
 * This can be done with an instance of the "onnx_torch::ModelProto" class.
 * To use an object of this class to lad the model, we need to use the "ParseFromIsttream" method, which takes the "std::istream" object as an input parameter.
 * The following code shows how to use an object of the "onnx_torch::ModelProto" class:*/
onnx_torch::ModelProto model_proto; 
{
	std::ifstream file(argv[1], std::ios_base::binary); 
	if (!file)
	{
		std::cerr << "File " << argv[1] << "can't be opened\n"; 
		return 1; 
	}

	if (!model_proto.ParseFromIstream(&file))
	{
		std::cerr << "Failed to parse onnx model\n"; 
		return 1; 
	}
}

/*Below code shows how to use the "SerializeToString" method of the "onnx_torch::ModelProto" class to make the model's string representation before we prepare the model:*/
std::string model_str; 
if (model_proto.SerializeToString(&model_str))
{
	caffe2::onnx::Caffe2Backend onnx_backend;	/*"caffe2::onnx::Caffe2Backend" class should be used to convert the Protobuf ONNX model into an internal representation of Caffe2.*/ 
	std::vector<caffe2::onnx::Caffe20ps> ops; 
	auto model = onnx_backed.Perpare(model_str, "CPU", ops); 	/*"Prepare" method, which takes the Protobuf formatted string, along with the model's description, a string containing the name of the conputational device, and some additional settings */
	if (model != nullptr)
	{
		...
	}
}

/*Now we have to prepare input and output data containers. 
 * The input is a tensor of size "1 x 3 x 224 x 224", which represents the RGB image for classification.
 * Now we need to move our image to the "inputs" vector.
 * Caffe2 tensor objects are not copyable, but they can be moved.
 * The "outputs" vector should be empty.*/

/*Below is a part of how  to prepare the input and output data for the model:*/
caffe2::TensorCPU image = ReadImageTensor(argv[2], 224, 224); 

std::vector<caffe2::TensorCPU> inputs; 
inputs.push_back(std::move(image));

std::vector<caffe2::TensorCPU> outputs(1); 

/*The model is an object of the "Caffe2BackendRep" class, which uses the "Run" method for evaluation. 
 * We can use it in the following way:*/
model->Run(inputs, &outputs); 

/*The output of this model is image scores for each of the 1000 classes of the ImageNet dataset, which was used to train the model. 
 * Below shows how to decode the model's output:*/
std::map<size_t, std::string> classes = ReadClasses(argv[3]); 
for (auto& output : outputs)
{
	const auto& probabilities = output.data<float>(); 
	std::vector<std::pair<float, int>> pairs; 		/*prob : class inded*/
	for (auto i = 0; i < output.size(); i++)
	{
		if (probabilities[i] > 0.01f)
		{
			pairs.push_back(std::make_pair(probabilities[i], i + 1)); /*0 - background*/
		}
	}

	std::sort(pairs.begain(), pairs.end()); 
	std::reverse(pairs.begain(), pairs.end()); 
	pairs.resize(std::min(5UL, pairs.size())); 
	for (auto& p : pairs)
	{
		std::cout << "Class " << p.second << " Label " << classes[static_cast<size_t>(p.second)] << " Prob " << p.first << std::endl; 
	}
	
	/*To correctly finish the program, we have to shut donw the Google "probtobuf" library, which we used to lead the required ONNX files:*/
	google::protobuf::ShutdonwProtobufLibrary(); 

}
