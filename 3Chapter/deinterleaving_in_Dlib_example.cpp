/*"Dlib" stores pixel in the row-major order with interleaved channels and data is placed in memory continuously with a single block. 
 * We can use raw pixel data to manage color values manually with the two function: 
 * 			1) "image_data()" to access raw pixel data
 * 			2) "width_step()" function to get the padding value*/

/*First step, we define containers for each of the channels*/
auto channel_size = static_cast<size_t>(img.nc() * img.nr()); 
std::vector<unsigned char> ch1(channel_size); 
std::vector<unsigned char> ch2(channel_size);
std::vector<unsigned char> ch3(channel_size); 

/*Next, we read color values for each pixel with two nested loops over image rows and columns.*/
size_t i{0}; 
for (long r = 0; r < img.nr(); ++r)
{
	for (long c = 0; c < img.nc(); ++c)
	{
		ch1[i] = img[r][c].red; 
		ch2[i] = img[r][c].green; 
		ch3[i] = img[r][c].blue; 
		++i; 
	}
}

/*The result is three containers with color channel values, which we can use separately. 
 * They are suitable to initialize grayscalled images for use in the image processing routines. 
 * We can use them to initialize a matrix-type object that we can process with linear algebra routines.*/
