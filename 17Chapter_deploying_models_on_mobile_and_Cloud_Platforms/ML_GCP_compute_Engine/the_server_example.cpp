/*The core of our application is the server. 
 * Let's assume that we've already implemented a image classification. 
 * That is by using a model saved as a TorchScript snapshot and loaded into the "torch::jit::script::Module" object.*/

/*Below phragment of code reads the parameters reuired by our application upon startup. 
 * There are three requried parameters: 
 * 	1) THe path to the model snapshot file
 * 	2) The path to the synset file,
 * 	3) the path to the directory when we placed our HTML client application files. 
 * 
 * There are also two optional paramters: 
 * 	1) the server host IP address 
 * 	2) the server network port */
class Network
{
	public: 
		Network(const std::string& snapshot_path, const std:string& synset_path, torch::DeviceType device_type); 

		std::string Classify(const at::Tensor& image); 

	private: 
		torch::DeviceType device_type_; 
		Classes classes_; 
		torch::jit::script::Module model_; 
}; 

/*After we've read the program parameters, we can initialize the "Network" type object with a specified model snapshot and synset files. 
 * We also dynamically determined whether there is a CUDA device available on the machine where we start the server. 
 * we did this with the "torch::cuda::is_available()" function.*/

/*The "torch::jit::load()" function accepts the device type as its second parameter and automatically moves the model to the specified device.
 * if a Cuda device is availale, we can move our model to this device to increase computational performance. 
 * Below code shows how we can load a model int oa specified device:*/
model_ = torch::jit::load(snapshot_path, device_type); 

/*There is a lightweight C++ single-file header-only cross-platform HTTP/HTTPS library available named "cpphttplib". 
 * We can use it to implement our server. 
 * Below code shows how we used the "httplib::Server" type to instanctiate the "server" object so that it can handle HTTP reuests:*/
httplib::Server server; 

/*The "httplib::Server" class also implements a simple static file server. 
 * Below shows how to set up the directory for loading static pages:*/
server.set_nase_dir(www_path.c_str()); 

/*The path that's passed into the "set_base_dir()" method should point to the directory we use to store the HTML pages for our service. 
 * TO be able to see what's going on in the server when it's launched, we can configure the logging function. 
 * Below code shows how to print minimal request information when the server accepts the incoming message:*/
server.set_logger([](const auto&)
		{
		std::cout << req.method << "\n" << std::endl; 
		}); 

/*It is also able to handle HTTP errors when our server works. 
 * The following code show how to fill the response object with error status information:*/
server.set_error_handler([](const auto&, auto& res)
		{
			std::stringstream buf; 
			buf << "<p>Error Status: <span style='color:red;'>"; 
			buff << res.status; 
			buff << "</span></p>"; 
			res.set_content(buf.str(), "text/html"); 
		}); 

/*Now, we have to configure the handler for our server object so that it can handle "Post" requests. 
 * There is a "Post" method in the "httplib::Server" class that we can use for this purpose. 
 * This method takes the name of the request's pattern and the handler object.*/

/*The special URL pattern should be used by the client application to perform a request; for example, the address can look like "http://localhost:6060/imgclassify", where "imgclassify" is the pattern. 
 * We can have different handlers for different request.
 * The handler can be any callale object that accepts two arguments: 
 * 	1) the first should be of the "const Request" type
 * 	2) should be of the "Response&" type. 
 * 
 * below code shows our implementation of the image classification request:*/
server.Post("/imgclassify", [&](const auto& req, auto& res))
{
	std::string response_string; 
	for (auto& file : req.files)
	{
		auto body = req.body.substr(file.second.offset, file.second.length); 
		try
		{
			auto img = RadMomoryImageTensor(body, 224, 224); 
			response_string += "; " + network.Classify(img); 
		} catch (...)
		{
			response_string += "; Classification failed"; 
		}
	}
	res.set_content(response_string.c_str(), "text/html"); 
}

/*In this handler, we iterated over all the files in the input request. 
 * For each file, we performed the following steps: 
 *
 * 	1) Extracted the bytes representing the image 
 * 	
 * 	2) Decoded the bytes into the image object
 *
 * 	3) Converted the image object into a tensor 
 *
 * 	4) Classified the image*/
/*We can use the "Listen()" method of the "httplib::Server" type object to enable it to accept incoming connections and processing messages. */
if(!server.listen(host.c_str(), port))
{
	std::cerr << "Failed to start server\n"; 
}/*The "listen()" method automatically binds the server socket to the given IP address and the port number.*/
