/*Consider the "MNISTDataset" class, which provides access to the MNIST dataset. 
 * The constructor of this class takes two parameters: 
 * 	1) The name of the file contains images 
 *
 * 	2) The name of the file that contains the labels
 *
 * It loads whole files into its memory, which is not a best practice, but for this dataset, this approach works well because the dataset is small. 
 * For bigger datasets, we have to implement another scheme of reading data from the disk because usually, for real tasks, we are unable to load all the data into the computer's memory. 
 *
 * We use the "OpenCV" library to deal with images, so we store all the loaded images in the C++ "vector" of the "cv::Mat" type. 
 * Lables are stored in a vector of the "unsigned char" type. 
 * We write two additional helper functions to read images and labels from the disk: "ReadImages" and "ReadLabels". 
 * below show a example of the code*/
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>

class MNISTDataset : public torch::data::Dataset<MNISTDataset>
{
	public : MNISTDataset(const std::string& images_file_name, const std::string& labels_file_name); 

		 /*torch::data::Dataset implementation*/
		 torch::data::Example<> get(size_t index) override; 
		 torch::optional<size_t> size() const override; 

	private: 
		 void ReadLabels(const std::string& labels_file_name); 
		 void ReadImages(const std::string& images_file_name); 

		 uint32_t rows_ - 0; 
		 uint32_t columns_ = 0; 
		 std::vector<unsigned char> labels_; 
		 std::vector<cv::Mat> images_; 
}; 

/*Next, is the implementation of the public interface of the class*/
MNISTDataset::MNISTDataset(const std::string& impages_file_name, const std::string& labels_file_name)
{
	ReadLabels(labels_file_name); 
	ReadImages(images_file_name); 
}

/*We can see that consttructor passed the filenames to the corresponding loader functions. 
 * The "size" method returns the number of items that were loaded from the disk into the labels container: */
torch::optional<size_t> MNISTDataset::size() const
{
	return labels_.size(); 
}

/*below is the "get" methods implementation*/
torch::data::Example<> MNISTDataset::get(size_t inded)
{
	return {CvImageToTensor(images_[index]), torch::tensor(static_cast<int64_t>(labels_[index]), torch::TensorOptions().dtype(torch::kLong).device(torch::DeviceType::CUDA))}; 
}

/*The "get" from "torch::data::Example<>" holds two values: 
 * 	1) The training sample represented with the "torch::Tensor" type
 *
 * 	2) target valure, which is also represented with the "torch::Tensor" type. 
 * This method retrieves an image from the corresponding container using a given subscript, converts the image into the "torch::Tensor" type with the "CvImageToTensor" function, and uses the label value converted into the "torch::Tensor" type as a target value*/
/*There is a set of "torch::tensor" function that are used to convert a C++ variable into the "torch::Tensor" type. 
 * They automatically deduce the variable type and create a tensor with corresponding values. 
 * We explicitly convert the label int the "int64_t" type because the loss function we'll be using later assumes that the target values have a "torch::Long" type. 
 * Also, notice that we passed "torch::TenserOptions" as a second argument to the "torch::tensor" function. 
 * We specified the torch type of the tensor values and told the system to place this tensor to the GPU memory by setting the "device" option on "torch::DeviceType::CUDA" and by configure where to place them -- in the CPU or GPU. 
 * Tensors that are placed in different types of memory can't be used together.
 *
 * To convert the OpenCV image into a tensor, below is the correct method*/
torch::Tensor CvImageToTensor(const cv::Mat& image)
{
	assert(image.channels() == 1); 

	std::vector<int64_t> dims{static_cast<int64_t>(1), static_cast<int64_t>(image.rows), static_cast<int64_t>(image.cols)}; 

	torch::Tensor tenser_image = torch::from_blob(image.data, torch::IntArrayRef(dims), torch::TensorOptions().dtype(torch::kFloat).requires_grad(false)).clone(); 		/*clone is required to copy data from temporary object*/
	return tensor_image.to(torch::DeviceType::CUDA); 
}

/*The most important part of this function is the call to the "torch::from_blob" function. 
 * This function constructs the tensor from values located in memory that are referenced by the pointer that's passed as a first argument. 
 * A second argument should be a C++ vector with tensor dimensions values; in our case, we specified a three-dimensional tensor with one channel and two image dimensions. 
 * The third argument is the "torch::TensorOptions" object. 
 * We specified that the data should be of the floating-point type and that it doesn't require a gradient calculation. 
 *
 * The third intresting PyTorch function that's used here is the "torch::Tensor::to" function, which allows us to move tensors from CPU memory to GPU memory and back */
