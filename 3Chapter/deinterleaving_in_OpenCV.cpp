/*When we load an image with the "OpenCV" it loads the image in the BGR format and with "char" as the underlying data type. 
 * So we need to convert it to the RGB format*/
cv::cvtColor(img, img, cv::COLOR_BGR2RGB); 

/*Next, we can convert the underlying data type to the "float" type*/
img.convertTO(img, CV_32FC3, 1/255.0); 

/*Next, to deinterleave channels, we need to split them with the "cv::split()" function. */
cv::Mat bgr[3]; 
cv::split(img, bgr); 

/*Last, we can place channels back to the "cv::Mat" object in the order we need with the "cv::vconcat()" function, which concatenates matrices vertically.*/
cv::Mat ordered_channels; 
cv::vconcat(bgr[2], bgr[1], ordered_channels); 
cv::vconcat(ordered_channels, bgr[0], ordered_channels); 
