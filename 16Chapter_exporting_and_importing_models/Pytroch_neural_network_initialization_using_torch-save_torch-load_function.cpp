/*"torch::save" function, which recursively saves parameters from the passed module:*/
torch::save(model, "pytorch_net.pt"); 

/*To use it correctly with our custom modules, we need to register all the submodules in the parent one with "register_module" module's method.*/
/*To load the save parameter, we can use the "torch::load" function:*/
Net model_loaded; 
torch::load(model_load, "pytorch_net.pt"); 
/*The function fills the passed module parameters with the vaues that are read from a file.*/
