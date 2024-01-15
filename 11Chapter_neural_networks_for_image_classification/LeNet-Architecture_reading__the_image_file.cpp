/*Reading the images file is also pretty straighforward: we read the header records and sequentially read the images. 
 * From the header records, we get the total number of images in the file sna the image size. 
 * Then, we define the OpenCV matrix object that has a corresponding size and type  -- the one channel image with the underlying byte "CV_8UC1" type. 
 * We read images from disk in a loop directly to the OpenCV matrix object by passing a pointer, which is returned by the "data" object variable, to the stream read function. 
 * The size of the data we need to read is determinded by calling the "cv::Mat::size()" function, followed by the call to the "area" function. 
 * Then, we use the "convertTo" OpenCV function to convert an image from "unsigned byte" type into 32 bit floating point type. 
 * This is important so that we have enough precisiion while performing math operations in the network layers. 
 * we also normalize all the data so that its in the range [0,1] by dividing by 255*/

/*We resize all the images so that they're 32 x 32 in size because the LeNet5 network architecture requires us to hold the original dimensions of the convolution filters: */
void MNISTDataset::ReadImages(const std::string& images_file_name)
{
	std::ifstream images_file(images_file_name, std::ios::binary | std::ios::binary); 
	labels_file.exceptions(std::ifstream::failbit | std::ifstream::badbit); 
	if (labels_file)
	{
		uint32_t magic_num = 0; 
		uint32_t num_items = 0; 
		rows_ = 0; 
		columns_ = 0; 
		if (read_header(&magic_num, labels_file) && read_header(&num_items, labels_file) && read_header(&rows_, labels_file) && reader_header(&columns_, labels_file))
		{
			assert(num_items == labels_.size()); 
			images_.resize(num_items); 
			cv::Mat img(static_cast<int>(rows_), static_cast<int>(columns_), CV_8UC1); 

			for (uint32_t i = 0; i < num_items; ++i)
			{
				images_file.read(reinterpret_cast<char*>(img.data), static_cast<std::streamsize>(img.size().area())); 
				img.convertTo(images_[i], CV_32F); 
				images_[i] /= 255; 	/*normalize*/
				cv::resize(images_[i], images_[i], cv::Size(32, 32)); 	/*Resize to 32 x 32 size*/
			}
		}
	}
}
