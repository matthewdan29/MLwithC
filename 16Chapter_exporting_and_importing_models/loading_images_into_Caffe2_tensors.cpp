/*Let's assume that we have the following function definition for image loading:*/
caffe2::TensorCPU ReadImageTensor(const std::string& file_name, int width, int height)
{
	...
}

/*Let's write its implementation.
 * For image loading, we will use the OpenCV library:*/
/*load image*/
auto image = cv::imread(file_name, cv::IMREAD_COLOR); 

if (!image.cols || !image.rows)
{
	return {}; 
}

if (image.cols != width || image.rows != height)
{
	/*scale image to fit*/
	cv::Size scaled(std::max(height * image.cols / image.rows, width), std::max(height, width * image.rows / image.cols)); 
	cv::resize(image, image, scaled); 

	/*crop image to fit*/
	cv::Rect corp((image.cols - width) / 2, (image.rows - height) / 2, width, height); 
	image = image(crop); 
}

/*Then, we convert the image into the floating-point type and RGB format:*/
image.convertTo(image, CV_32FC3); 
cv::cvtColor(image, image, cv::COLOR_BGR2RGB); 

/*After formattiong the complete, we can split the image into three seprate channels with red, green, and blue, colors.
 * We should also normalize the color values.*/
std::vector<cv::Mat> channels(3); 
cv::split(image, channels); 

std::vector<double> mean = {0.485, 0.456. 0.406}; 
std::vector<double> stddev = {0.229, 0.224, 0.225}; 

size_t i = 0; 
for (auto& c : channels)
{
	c = ((c / 255) - mean[i]) / stdev[i]; 
	++i; 
}

/*Then, we should concatenate the channels:*/
cv::vconcat(channels[0], channels[1], image); 	/*"cv::vconcat" function normalized chnnels we concatenated into one contiguous image*/
cv::vconcat(image, channels[2], image); 	/*"cv::vconcat" function normalized channels we concatenated int one contiguous image (do i smell CI/CD...)*/
assert(image.isConntinuous()); 

/*Below shows how to initialize the Caffe2 tensor with the image data:*/
std::vector<int64_t> dims = {1, 3, height, width}; 

caffe2::TensorCPU tensor(dims, caffe2::DeviceType::CPU);	/*"caffe2::TensorCPU" object copies the image data which was initialized with the specified dimensions.*/
std::copy_n(reinterpret_cast<float*>(image.data), image.size().area(), tensor.mutable_data<float>());	/*"mutable_data<float>()" member function to access the internal storage of the tensor.*/

return tensor; 


