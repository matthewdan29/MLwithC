/*This library has different types used for math routines and image processing. 
 * "Dlib::array2d" is recommended for types of images. 
 * "Dlib::array2d" type is a template type that has to be parametrized with a pixel type. 
 * Here are the following predefined pixel types: 
 * 		1) "rgb_pixel"
 * 		2) "bgr_pixel"
 * 		3) "rgb_alpha_pixel"
 * 		4) "hsi_pixel"
 * 		5) "lab_pixel"
 * Also any scalar type can be used for the grayscaled pixels representation.*/

/*We can use the "load_image()" function to load an image from disk*/
#include <Dlib/image_io.h>
#include <Dlib/image_transforms.h>
using namespace Dlib; 
...
array2d<rgb_pixel> img; 
load_image(img, file_path); 

/*"Dlib::resize_image()" function is used for scaling operations. 
 * The function has 2 different overloads: 
 * 		1) One takes a single scale factor and a reference to an image object. 
 * 		2) The second one takes an input image, and output image, the desired size, and an interpolation type.
 * Use "interpolate_nearest_neighber()", "interpolate_qudratic()", "interplate_bilinear()" functions to specify the interpolation type in the "Dlib" library*/
array2d<rgb_pixel> img2(img.nr() / 2, img.nc() / 2); 
resize_image(img, img2, interpolate_nearest_neighbor()); 
resize_image(1.5, img); 				/*default interpolate_bilinear*/

/*"Dlib::extract_image_chips()" is used to crop. 
 * This function takes an original image, rectangle-defined bounds, and an output image, 
 * There are overloads of this function that take an array of rectangle bounds and an array of output images*/
extract_image_chip(img, rectangle(0, 0, img,nc() /2, img,nr() /2), img2); 

/*"Dlib::transform_image()" function which takes an input image, and output image, and an affine transformation object to support image transformations operations. 
 * An example of the transformation object could be an instance of the "Dlib::point_transform_affine" classs, which defines the affine transformation with a rotation matrix and an translation vecotr. 
 * "Dlib::transform_image()" function can take an interpolation type as the last parameter*/
transform_image(img, img2, interpolate_bilinear(), point_transform_affine(identity_matrix<double, 2>(-50, -50))); 

/*For only roations, "Dlib::rotate_image()" function is you method of choice. 
 * "Dlib::rotate_image()" function takes an input image, an output image, a rotation angle in degrees, and interpolation type*/
rotat_image(img, img2, -45, interpolate_bilinear()); 

/*There are two methods for adding borders in the "Dlib" there are two methods: 
 * 		1) "Dlib::assign_border_pixels()"
 * 		2) "Dlib::zero_border_pixels()"
 * For filling image borders with specified values. 
 * Befor using these routines, we should resize the image and place the context in the right position. 
 * Thie new image size should include borders' width. 
 * We can use the "Dlib::transform_image()" function to move the image content into the right place. */
int top = 50; 		//px
int bottom = 20; 	//px
int left = 150; 	//px
int right = 5; 		//px
img2.set_size(img.nr() + top + bottom, img.nc() + left + right); 
transform_image(img, img2, interpolate_bilinear(), point_transform_affine(identity_matrix<double>(2), Dlib::vector<double, 2>(-left/2, -top/2))); 

/*To convert an image to another color space, we should define a new image with the desired type of pixels and pass it to the "Dlib::assign_image()" function. 
 * below example show how to convert the RGB image to a blue, green, red (BGR) one*/
array2d<bgr_pixel> img_brg; 
assign_image(img_bgr, img); 

/*To make a grayscale image, we can define an image with the "unsigned char" pixel type.*/
array2d<unsigned char> img_gray; 
assign_image(img_gray, img); 
