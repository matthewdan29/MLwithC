ONNX format is a special file format used to share neural network architectures and parameters between different frameworks. 
It is based on the Google Protobuf format and library. 
The reason why this format exists is to test and run the same neural network model in different environments and on different devices. 
Usually, researcher use a programming framework that they know how to use in order to develop a model, and then run this model in a developer. 
This format is supported by all leading frameworks, such as "PyTorch", "TensorFlow", "MXNet", and others. 
But now, there is a lack of support for this format from the C++ API of these frameworks and atthe time of writting, they only have a python interface for dealing with ONNX format. 
Some time ago, Facebook developed the Caffe2 neural network framework in order to run models on different platforms with the best performance. 
This framework also had a C++ API, and it was able to load and run models save in ONNX format. 
Now, this framework had been merged with "PyTorch". 
There is a plan to remove the Caffe2 api and replace it with a new combined API in "PyTorch". 
But at the time of writing, the Caffe2 C++ API is still available as part of the "PyTorch" 1.2 (libtorch) library. 

