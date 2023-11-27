/*"OpenCV" and image is treated as a multidimensional matrix of values. 
 * There is a special "cv::Mat" type for this purpose. 
 * There are two base functions: 
 * 		1) "cv::imread()" function loads the image
 * 		2) "cv::imwrite()" function writes the image to a file*/
#include <openscv2/opencv.hpp>
...
cv::Mat img = cv::imread(file_name); 
cv::imwrite(new_file_name, img); 

/*There are functions to manage images located in a memory buffer. 
 * The "cv::imdecode()" function loads an image from the memory buffer and the "cv""imencode()" function writes an image to memory buffer.*/

/*Scaling operations in the "OpenCV" library can be donw with the "cv::resize()" function. 
 * This function takes an input image, and output image, the output image size or scale factors, and an interpolation type as arguments. 
 * The interpolation type governs how the output image will look after the scaling. */
cv::resize(img, img, {img.cols / 2, img,rows / 2}, 0, 0, cv::INTER_AREA); 
cv::resize(img, img, {}, 1.5, 1.5, cv::INTER_CUBIC); 

/*The "cv::Mat" type overrides the "operator()" method, which takes a cropping rectangle as an argument and returns a new "cv::Mat" object with part of the image surrounded by the specified rectangele. 
 * Note that his object will share the same memory with the original image, so its modification will change the original image too. 
 * To make a deep copy of the "cv::Mat" object we need to use the "clone()" method*/
img = img(cv::Rect(0, 0, img.cols / 2, img.rows / 2)); 

/*"OpenCV" supports translation and rotation operations. 
 * You have to create a matrix of 2D affine transformations and then apply it to our image. 
 * We can create such a matrix manually, and then apply it to an image with the "cv::wrapAffine()" function*/
cv::Mat trm = (cv::Mat_<double>(2,3) << 1, 0, -50); 
cv::wrapAffine(img, img, trm, {img.cols, img.rows}); 

/*We can create a rotation matrix with the "cv::getRotationMatrix2D()" function. 
 * This takes a point of origin and the rotation angle in degrees*/
auto rotm = cv::getRotationMatrix2D({img.cols / 2, img.rows / 2}, 45, 1); 
cv::wrapAffine(img, img, rotm, {img.cols, img.rows}); 

/*Another useful operation is extending an image size without scaling but adding borders. 
 * There is the "cv::copyMakeBorder()" function in the "OpenCV" lib for this purpose. 
 * This function has different options on how to creaate borders. 
 * It takes an input image, and output image, norder size for the top, the bottom, the left and right sides, types of border, and border color.*/
int top = 50; 
int bottom = 20; 
int left = 150; 
int right = 5; 
cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT | cv::BORDER_ISOLOATED, cv::Scalar(255, 0, 0)); 

/*With "cv::copyMakeBorder()" is very helpful when we need to adapt training images of different sizes to the one standard image size used in some machine learning algorithms because, with this function we do not distort target image content*/

/*There is the "cv::cvtColor()" function to convert different color spaces in the "OpenCV" lib. 
 * The function takes an input image, and output image, and an conversion scheme type. */
cv::cvtColor(img, img, cv::COLOR_RGB2GRAY); /*now pixels values are in range of 0 - 1*/


