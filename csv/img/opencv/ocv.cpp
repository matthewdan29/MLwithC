#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem; 

int main(int argc, char** argv)
{
	cv::Mat img; 
	if (argc > 1)
	{
		auto file_path = fs::path(argv[1]); 
		if (fs::exists(file_path))
		{
			cv::Mat img = cv::imread(file_path.string()); 
		} else 
		{
			std::cerr << "File path " << file_path << " is invalid\n"; 
		}
	}

	if (img.empty())
	{
		const int width = 512; 
		img = cv::Mat(width, width, CV_32FC3); 
		img = cv::Scalar(255, 255, 255); 
		cv::rectangle(img, {width / 4, width / 4}, {width / 2, width / 2}, cv::Scalar(0, 0, 0), -1); 	/*negative thickness fill rect*/
	}
	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE); 	/*create a window for display*/
	cv::imshow("Image", img); 			/*show our image inside it.*/
	cv::waitKey(0); 

	/*Scalling*/
	/*use cv::INTER_AREA for shrinking */
	/*cv::INTER_CUBIC (slow) or cv::INTER_LINEAR for zooming. */
	/*cv::INTER_LINEAR is used for all resizing purposes*/
	cv::resize(img, img, {img.cols / 2, img.rows /2}, 0, 0, cv::INTER_AREA);
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	cv::resize(img, img, {}, 1.5, 1.5, cv::INTER_CUBIC); 
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*cropping*/
	img = img(cv::Rect(0, 0, img.cols / 2, img.rows / 2)); 
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*translation*/
	cv::Mat trm = (cv::Mat_<double>(2, 3) << 1, 0, -50, 0, 1, -50); 
	cv::warpAffine(img, img, trm, {img.cols, img.rows}); 
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*rotations*/
	auto rotm = cv::getRotationMatrix2D({img.cols / 2, img.rows / 2}, 45, 1); 
	cv::warpAffine(img, img, rotm, {img.cols, img.rows}); 
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*padding*/
	/*types of orders:*/
	/*BORDER_CONSTANT - single color*/
	/*BORDER_REPLICATE - copy last values - aaaa|abcdefgh|hhhh*/
	/*BORDER_REFLECT - copy opossite image values */
	/*BORDER_WRAP - simulate image duplication*/
	/*From doc: When the source image is a part (ROI) of a bigger image, the function will try to use the pixels outside of the ROI to from a border.
	 * To disable this feature and always do extrapolation, as if src was not a ROI, use borderType | BORDER_ISOLATED.*/
	int top = 50; 	/*px*/
	int bottom = 20; 	/*px*/
	int left 150; 		/*px*/
	int right = 5; 		/*px*/

	cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(255, 0, 0)); 
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*Convert from default BGR to rgb*/
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB); 
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*make grayscale*/
	cv::cvtColor(img, img, cv::COLOR_RGB2GRAY); 	/*now pixels values are in range 0 - 1*/
	std::cout << "Grayscale image channels " << img.channels() << std::endl;
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*Change underlaying data type*/
	img.converTo(img, CV_8UC1, 255); 	/*float -> byte*/
	cv::imshow("Image", img); 
	cv::waitKey(0); 

	/*Mix layers*/
	/*layout of channels i in memory n OpenCV can be non continuous*/
	/*Also, interleaved, so usually before passing OpenCV image to another library we mix to restructre them*/
	img = cv::Mat(512, 512, CV_32FC3); 
	img = cv::Scalar(255, 255, 255); 
	cv::split(img, bgr); 
	cv::Mat ordered_channels; 
	cv::vconcat(bgr[2], bgr[1], ordered_channels); 
	cv::vconcat(ordered_channels, bgr[0], ordered_channels); 

	std::cout << "Memory layout is continuous " << ordered_channels.isContinuous() << std::endl; 

	return 0; 
}
