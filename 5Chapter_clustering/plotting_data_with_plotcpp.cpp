...
using Coords = std::vector<DataType>; 
using PointCoords = std::pair<Coords, Coords>; 
using Clusters = std::unordered_map<index_t, PointCoords>; 

const std::vector<std::string> colors{"black", "red", "blue", "green", "cyan", "yellow", "brown", "megenta"}; 
...
void PlotClusters(const Clusters& clusters, const std::string& name, const std::string& file_name)
{
	plotcpp::Plot plt; 
	plt.SetTerminal("png"); 
	plt.SetOutput(file_name); 
	plt.SetTitle(name); 
	plt.SetXLabel("x"); 
	plt.SetYLabel("y"); 
	plt.SetAutoscale(); 
	plt.gnuplotCommand("set grid"); 

	auto draw_state = plt.StartDraw2D<Coords::const_iterator>(); 
	for (auto& cluster : clusters)
	{
		std::stringstream params; 
		params << "lc rgb '" << colors[cluster.first] << "' pt 7"; 
		plt.AddDrawing(draw_state, plotcpp::Points(cluster.second.first.begin(), cluster.second.first.end(), cluster.second.second.begin(), std::to_string(cluster.first) + " cls", params.str())); 
	}
	plt.EndDraw2D(draw_state); 
	plt.Flush(); 
}
