/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% driver for program fem %%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% %%
 %% Variables: %%
 %% nruns = number of program runs %%
 %% xpts = xpoints or nodes for fem %%
 %% n = number of nodes - 1 %%
 %% el2 = l-2 norm error %%
 %% emax = infinity norm error %%
 %% u = vector containing the n+1 basis coefficients - %%
 %% a.k.a. the fem solution %%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% %%
 %% Calls the functions assem and bookkeep to generate %%
 %% the linear system to be solved using PETSc KSP methods. %%
 %% Prints the output to a file named output.txt %%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "mycalls.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include <time.h>
#include "petscksp.h"
#include "petscmat.h"
#include <mpi.h>

class Constants{
public:
    static double kLengthOfSquareDomain_;
    static int kNumOfHeaters_;
    static double kStefanBoltzmann_;
    static int kNumOfNodesInElement_;
    static int kNumOfDofsPerNode_;
    static int kNumOfMaterials_;
    static double kNormTolerance_;
    static double kYFunctionTolerance_;
    static int kMaxNewtonIteration_;
    static int kMeshSeedsAlongSiliconThickness_;
    static int kMeshSeedsAlongSTitaniumThickness_;
    static double kMinYCoordinate_;
};
double Constants::kLengthOfSquareDomain_ = 1.0;
int Constants::kNumOfHeaters_ = 10;
double Constants::kStefanBoltzmann_=5.6703e-11; // units mW/(mm^2 * K^4)   5.6703e-8 W*m^-2*K^-4
int Constants::kNumOfNodesInElement_=4;
int Constants::kNumOfDofsPerNode_=1;
int Constants::kNumOfMaterials_=4;
double Constants::kNormTolerance_=1.0e-5;
double Constants::kYFunctionTolerance_=1.0e-5;
int Constants::kMaxNewtonIteration_=10;
int Constants::kMeshSeedsAlongSiliconThickness_=5;
int Constants::kMeshSeedsAlongSTitaniumThickness_=1;
double Constants::kMinYCoordinate_=0.0;


class ModelGeometry{
public:
    void set_length_of_model()
    {length_of_model_=Constants::kLengthOfSquareDomain_+2*width_of_end_;}
    void set_thickness_of_csilicon(const double thickness_of_csilicon)
    {thickness_of_csilicon_=thickness_of_csilicon;}
    void set_thickness_of_titanium(const double thickness_of_titanium)
    {thickness_of_titanium_=thickness_of_titanium;}
    void set_thickness_of_isolater(const double thickness_of_isolater)
    {thickness_of_isolater_=thickness_of_isolater;}
    void set_thickness_of_silicondioxide(const double thickness_of_silicondioxide)
    {thickness_of_silicondioxide_=thickness_of_silicondioxide;}
    void set_thickness_of_copper(const double thickness_of_copper)
    {thickness_of_copper_=thickness_of_copper;}
    void set_width_of_end(const double width_of_end)
    {width_of_end_=width_of_end;}
    void set_new_width_of_end(const double width_of_end)
    {width_of_end_=width_of_end;}
    void set_width_of_heater(const double width_of_heater)
    {width_of_heater_=width_of_heater;}
    void set_width_of_gap()
    {width_of_gap_= Constants::kLengthOfSquareDomain_/Constants::kNumOfHeaters_ - width_of_heater_;}
    double get_length_of_model() const
    {return length_of_model_;}
    double get_thickness_of_csilicon() const
    {return thickness_of_csilicon_;}
    double get_thickness_of_titanium() const
    {return thickness_of_titanium_;}
    double get_thickness_of_isolater() const
    {return thickness_of_isolater_;}
    double get_thickness_of_silicondioxide() const
    {return thickness_of_silicondioxide_;}
    double get_thickness_of_copper() const
    {return thickness_of_copper_;}
    double get_width_of_end() const
    {return width_of_end_;}
    double get_width_of_heater() const
    {return width_of_heater_;}
    double get_width_of_gap() const
    {return width_of_gap_;}
    
private:
    double length_of_model_;
    double thickness_of_csilicon_;
    double thickness_of_titanium_;
    double thickness_of_isolater_;
    double thickness_of_silicondioxide_;
    double thickness_of_copper_;
    double width_of_end_;
    double width_of_heater_;
    double width_of_gap_;
};


class AnalysisConstants{
public:
    void set_ambient_temperature(const double ambient_temperature)
    {ambient_temperature_=ambient_temperature; }
    void set_boundary_condition_temperature(const double boundary_condition_temperature)
    {boundary_condition_temperature_=boundary_condition_temperature; }
    void set_output_time_step_interval(const int output_time_step_interval)
    {output_time_step_interval_=output_time_step_interval;}
    void set_sample_initial_temperature(const double sample_initial_temperature)
    {sample_initial_temperature_=sample_initial_temperature;}
    void set_maximum_time_steps(const int maximum_time_steps)
    {maximum_time_steps_=maximum_time_steps;}
    void set_total_simulation_time(const double total_simulation_time)
    {total_simulation_time_=total_simulation_time;}
    void set_initial_time_increment(const double initial_time_increment)
    {initial_time_increment_=initial_time_increment;}
    void set_minimum_time_increment(const double minimum_time_increment)
    {minimum_time_increment_=minimum_time_increment;}
    void set_time_to_turn_off_heaters(const double time_to_turn_off_heaters)
    {time_to_turn_off_heaters_=time_to_turn_off_heaters;}
    void set_maximum_temperature_change_per_time_increment(const double maximum_temperature_change_per_time_increment)
    {maximum_temperature_change_per_time_increment_=maximum_temperature_change_per_time_increment;}
    double get_boundary_condition_temperature() const
    {return boundary_condition_temperature_; }
    double get_ambient_temperature() const
    {return ambient_temperature_; }
    double get_time_to_turn_off_heaters() const
    {return time_to_turn_off_heaters_;}
    double get_output_time_step_interval() const
    {return output_time_step_interval_;}
    double get_sample_initial_temperature() const
    {return sample_initial_temperature_;}
    int get_maximum_time_steps() const
    {return maximum_time_steps_;}
    double get_total_simulation_time() const
    {return total_simulation_time_;}
    double get_initial_time_increment() const
    {return initial_time_increment_;}
    double get_minimum_time_increment() const
    {return minimum_time_increment_;}
    double get_maximum_temperature_change_per_time_increment() const
    {return maximum_temperature_change_per_time_increment_;}
    
private:
    double ambient_temperature_;
    double boundary_condition_temperature_;
    double time_to_turn_off_heaters_;
    int output_time_step_interval_;
    double sample_initial_temperature_;
    int maximum_time_steps_;
    double total_simulation_time_;
    double initial_time_increment_;
    double minimum_time_increment_;
    double maximum_temperature_change_per_time_increment_;
};


class MeshParameters{
public:
    void set_mesh_seeds_on_end(const int mesh_seeds_on_end)
    {mesh_seeds_on_end_=mesh_seeds_on_end;}
    void set_mesh_seeds_on_gap(const int mesh_seeds_on_gap)
    {mesh_seeds_on_gap_=mesh_seeds_on_gap;}
    void set_mesh_seeds_on_heater(const int mesh_seeds_on_heater)
    {mesh_seeds_on_heater_=mesh_seeds_on_heater;}
    void set_mesh_seeds_along_isolater_thickness(const int mesh_seeds_along_isolater_thickness)
    {mesh_seeds_along_isolater_thickness_=mesh_seeds_along_isolater_thickness;}
    void set_mesh_seeds_along_silicondioxide_thickness(const int mesh_seeds_along_silicondioxide_thickness)
    {mesh_seeds_along_silicondioxide_thickness_=mesh_seeds_along_silicondioxide_thickness;}
    void set_mesh_seeds_along_copper_thickness(const int mesh_seeds_along_copper_thickness)
    {mesh_seeds_along_copper_thickness_=mesh_seeds_along_copper_thickness;}
    int get_mesh_seeds_on_end() const
    {return mesh_seeds_on_end_;}
    int get_mesh_seeds_on_gap() const
    {return mesh_seeds_on_gap_;}
    int get_mesh_seeds_on_heater() const
    {return mesh_seeds_on_heater_;}
    int get_mesh_seeds_along_isolater_thickness() const
    {return mesh_seeds_along_isolater_thickness_;}
    int get_mesh_seeds_along_silicondioxide_thickness() const
    {return mesh_seeds_along_silicondioxide_thickness_;}
    int get_mesh_seeds_along_copper_thickness() const
    {return mesh_seeds_along_copper_thickness_;}
    void set_dimensions_of_x()
    {dimensions_of_x_=Constants::kNumOfHeaters_*(mesh_seeds_on_heater_+mesh_seeds_on_gap_)+2*mesh_seeds_on_end_+1;}
    int get_dimensions_of_x() const
    {return dimensions_of_x_;}
    void set_dimensions_of_y(){
        dimensions_of_y_ = Constants::kMeshSeedsAlongSiliconThickness_+Constants::kMeshSeedsAlongSTitaniumThickness_+
        mesh_seeds_along_silicondioxide_thickness_+mesh_seeds_along_isolater_thickness_+mesh_seeds_along_copper_thickness_+1;
    }
    int get_dimensions_of_y() const
    {return dimensions_of_y_;}
    int set_num_of_nodes()
    {num_of_nodes_=dimensions_of_x_*dimensions_of_y_;}
    int get_num_of_nodes() const
    {return num_of_nodes_;}
    int set_num_of_elements()
    {num_of_elements_=(dimensions_of_x_-1)*(dimensions_of_y_-1);}
    int get_num_of_elements() const
    {return num_of_elements_;}
    
private:
    int mesh_seeds_on_end_;
    int mesh_seeds_on_gap_;
    int mesh_seeds_on_heater_;
    int mesh_seeds_along_isolater_thickness_;
    int mesh_seeds_along_silicondioxide_thickness_;
    int mesh_seeds_along_copper_thickness_;
    int num_of_nodes_;
    int num_of_elements_;
    int dimensions_of_x_;
    int dimensions_of_y_;
};


class currentsInHeater{
public:
    void InitilizecurrentsInHeater();
    void set_current_in_heater(const int the_heater_num, const double current_in_heater);
    std::vector<double>& get_current_in_heater()
    {return currents_in_heater_;}
    
private:
    std::vector<double> currents_in_heater_;
};
void currentsInHeater::InitilizecurrentsInHeater(){
    currents_in_heater_.clear();
    currents_in_heater_.resize(Constants::kNumOfHeaters_,0.0);
}

void currentsInHeater::set_current_in_heater(const int the_heater_num, const double current_in_heater){
    currents_in_heater_[the_heater_num]=current_in_heater;
}


class ReadInput{
public:
    void ScanInputInformation();
    double get_time_to_turn_off_heaters() const
    {return time_to_turn_off_heaters_;}
    int get_mesh_seeds_on_end() const
    {return mesh_seeds_on_end_;}
    int get_mesh_seeds_on_gap() const
    {return mesh_seeds_on_gap_;}
    int get_mesh_seeds_on_heater() const
    {return mesh_seeds_on_heater_;}
    int get_mesh_seeds_along_isolater_thickness() const
    {return mesh_seeds_along_isolater_thickness_;}
    int get_mesh_seeds_along_silicondioxide_thickness() const
    {return mesh_seeds_along_silicondioxide_thickness_;}
    int get_mesh_seeds_along_copper_thickness() const
    {return mesh_seeds_along_copper_thickness_;}
    int get_output_time_step_interval() const
    {return output_time_step_interval_;}
    int get_maximum_time_steps() const
    {return maximum_time_steps_;}
    double get_initial_time_increment() const
    {return initial_time_increment_;}
    double get_minimum_time_increment() const
    {return minimum_time_increment_;}
    double get_width_of_end() const
    {return width_of_end_;}
    double get_thickness_of_copper() const
    {return thickness_of_copper_;}
    double get_thickness_of_csilicon() const
    {return thickness_of_csilicon_;}
    double get_thickness_of_isolater() const
    {return thickness_of_isolater_;}
    double get_thickness_of_silicondioxide() const
    {return thickness_of_silicondioxide_;}
    double get_thickness_of_titanium() const
    {return thickness_of_titanium_;}
    double get_total_simulation_time() const
    {return total_simulation_time_;}
    double get_width_of_heater() const
    {return width_of_heater_;}
    double get_ambient_temperature() const
    {return ambient_temperature_;}
    double get_boundary_condition_temperature() const
    {return boundary_condition_temperature_;}
    double get_sample_initial_temperature() const
    {return sample_initial_temperature_;}
    double get_maximum_temperature_change_per_time_increment() const
    {return maximum_temperature_change_per_time_increment_;}
    std::vector<double>& get_current_in_heater()
    {return currents_in_heater_;}
    
private:
    double time_to_turn_off_heaters_;
    int mesh_seeds_on_end_;
    int mesh_seeds_on_gap_;
    int mesh_seeds_on_heater_;
    int mesh_seeds_along_isolater_thickness_;
    int mesh_seeds_along_silicondioxide_thickness_;
    int mesh_seeds_along_copper_thickness_;
    int output_time_step_interval_;
    int maximum_time_steps_;
    double initial_time_increment_;
    double minimum_time_increment_;
    double ambient_temperature_;
    double boundary_condition_temperature_;
    double width_of_end_;
    double sample_initial_temperature_;
    double thickness_of_copper_;
    double thickness_of_csilicon_;
    double thickness_of_isolater_;
    double thickness_of_silicondioxide_;
    double thickness_of_titanium_;
    double total_simulation_time_;
    double width_of_heater_;
    double maximum_temperature_change_per_time_increment_;
    std::vector<double> currents_in_heater_;
};
void ReadInput::ScanInputInformation(){
    std::ifstream ifs("input.txt", std::ios::in);
    char ignore_this_string[80];
    ifs>>ignore_this_string;
    ifs>>mesh_seeds_on_end_;
    ifs>>ignore_this_string;
    ifs>>mesh_seeds_on_gap_;
    ifs>>ignore_this_string;
    ifs>>mesh_seeds_on_heater_;
    ifs>>ignore_this_string;
    ifs>>mesh_seeds_along_isolater_thickness_;
    ifs>>ignore_this_string;
    ifs>>mesh_seeds_along_silicondioxide_thickness_;
    ifs>>ignore_this_string;
    ifs>>mesh_seeds_along_copper_thickness_;
    ifs>>ignore_this_string;
    ifs>>time_to_turn_off_heaters_;
    ifs>>ignore_this_string;
    ifs>>output_time_step_interval_;
    ifs>>ignore_this_string;
    ifs>>maximum_time_steps_;
    ifs>>ignore_this_string;
    ifs>>total_simulation_time_;
    ifs>>ignore_this_string;
    ifs>>initial_time_increment_;
    ifs>>ignore_this_string;
    ifs>>minimum_time_increment_;
    ifs>>ignore_this_string;
    ifs>>maximum_temperature_change_per_time_increment_;
    ifs>>ignore_this_string;
    ifs>>width_of_end_;
    ifs>>ignore_this_string;
    ifs>>thickness_of_copper_;
    ifs>>ignore_this_string;
    ifs>>thickness_of_csilicon_;
    ifs>>ignore_this_string;
    ifs>>thickness_of_isolater_;
    ifs>>ignore_this_string;
    ifs>>thickness_of_silicondioxide_;
    ifs>>ignore_this_string;
    ifs>>thickness_of_titanium_;
    ifs>>ignore_this_string;
    ifs>>width_of_heater_;
    ifs>>ignore_this_string;
    ifs>>ambient_temperature_;
    ifs>>ignore_this_string;
    ifs>>boundary_condition_temperature_;
    ifs>>ignore_this_string;
    ifs>>sample_initial_temperature_;
    ifs>>ignore_this_string;
    
    currents_in_heater_.resize(Constants::kNumOfHeaters_, 0.0);
    for(int i=0;i<Constants::kNumOfHeaters_;i++){
        ifs>>currents_in_heater_[i];
        //    ifs>>ignore_this_string;
    }
    ifs.close();
}


// Class DisperseInputData disperse the data in Class ReadInput to the corresponding classified classes to store the data.
class Initialization{
public:
    void ScanInputInformation()
    {read_input_.ScanInputInformation();}
    void DeliverDataToAnalysisConstants();
    void DeliverDataToMeshParameters();
    void DeliverDataToModelGeometry();
    void DeliverDataTocurrentInHeater();
    void InitializeInitialization();
    ModelGeometry *const get_model_geometry()
    {return &model_geometry_;}
    AnalysisConstants *const get_analysis_constants()
    {return &analysis_constants_;}
    MeshParameters *const get_mesh_parameters()
    {return &mesh_parameters_;}
    currentsInHeater *const get_currents_in_heater()
    {return &currents_in_heater_;}
    
protected:
    ModelGeometry model_geometry_;
    AnalysisConstants analysis_constants_;
    MeshParameters mesh_parameters_;
    currentsInHeater currents_in_heater_;
    ReadInput read_input_;
};
void Initialization::InitializeInitialization(){
    ScanInputInformation();
    DeliverDataToAnalysisConstants();
    DeliverDataToMeshParameters();
    DeliverDataToModelGeometry();
    currents_in_heater_.InitilizecurrentsInHeater();
    DeliverDataTocurrentInHeater();
    mesh_parameters_.set_dimensions_of_x();
    mesh_parameters_.set_dimensions_of_y();
    mesh_parameters_.set_num_of_nodes();
    mesh_parameters_.set_num_of_elements();
}

void Initialization::DeliverDataToAnalysisConstants(){
    analysis_constants_.set_time_to_turn_off_heaters(read_input_.get_time_to_turn_off_heaters());
    analysis_constants_.set_output_time_step_interval(read_input_.get_output_time_step_interval());
    analysis_constants_.set_maximum_time_steps(read_input_.get_maximum_time_steps());
    analysis_constants_.set_total_simulation_time(read_input_.get_total_simulation_time());
    analysis_constants_.set_initial_time_increment(read_input_.get_initial_time_increment());
    analysis_constants_.set_minimum_time_increment(read_input_.get_minimum_time_increment());
    analysis_constants_.set_maximum_temperature_change_per_time_increment(read_input_.get_maximum_temperature_change_per_time_increment());
    analysis_constants_.set_ambient_temperature(read_input_.get_ambient_temperature());
    analysis_constants_.set_boundary_condition_temperature(read_input_.get_boundary_condition_temperature());
    analysis_constants_.set_sample_initial_temperature(read_input_.get_sample_initial_temperature());
}

void Initialization::DeliverDataToMeshParameters(){
    mesh_parameters_.set_mesh_seeds_on_end(read_input_.get_mesh_seeds_on_end());
    mesh_parameters_.set_mesh_seeds_on_gap(read_input_.get_mesh_seeds_on_gap());
    mesh_parameters_.set_mesh_seeds_on_heater(read_input_.get_mesh_seeds_on_heater());
    mesh_parameters_.set_mesh_seeds_along_isolater_thickness(read_input_.get_mesh_seeds_along_isolater_thickness());
    mesh_parameters_.set_mesh_seeds_along_silicondioxide_thickness(read_input_.get_mesh_seeds_along_silicondioxide_thickness());
    mesh_parameters_.set_mesh_seeds_along_copper_thickness(read_input_.get_mesh_seeds_along_copper_thickness());
}

void Initialization::DeliverDataToModelGeometry(){
    model_geometry_.set_thickness_of_copper(read_input_.get_thickness_of_copper());
    model_geometry_.set_thickness_of_csilicon(read_input_.get_thickness_of_csilicon());
    model_geometry_.set_thickness_of_isolater(read_input_.get_thickness_of_isolater());
    model_geometry_.set_thickness_of_silicondioxide(read_input_.get_thickness_of_silicondioxide());
    model_geometry_.set_thickness_of_titanium(read_input_.get_thickness_of_titanium());
    model_geometry_.set_width_of_heater(read_input_.get_width_of_heater());
    model_geometry_.set_width_of_gap();
    model_geometry_.set_width_of_end(read_input_.get_width_of_end());
    model_geometry_.set_length_of_model();
    model_geometry_.set_new_width_of_end(read_input_.get_width_of_end());
}

void Initialization::DeliverDataTocurrentInHeater(){
    for(int i=0;i<Constants::kNumOfHeaters_;i++){
        currents_in_heater_.set_current_in_heater(i, read_input_.get_current_in_heater()[i]);
    }
}


//class GenerateMesh is used to generate the mesh for analysis
class GenerateMesh{
public:
    void CalculateCoordinates(Initialization* const);
    std::vector<double> &get_x_coordinates()
    {return x_coordinates_;}
    std::vector<double> &get_y_coordinates()
    {return y_coordinates_;}
    void GenerateMeshInitializeMeshSizeInfo(Initialization* const);
    void PrintCoordinatesResults(Initialization* const);
    
private:
    std::vector<double> x_coordinates_;
    std::vector<double> y_coordinates_;
    std::vector<double> x_coordinates_candidates;
    std::vector<double> y_coordinates_candidates;
    double mesh_size_on_end_;
    double mesh_size_on_gap_;
    double mesh_size_on_heater_;
    double mesh_size_along_isolater_thickness_;
    double mesh_size_along_silicondioxide_thickness_;
    double mesh_size_along_copper_thickness_;
};

void GenerateMesh::GenerateMeshInitializeMeshSizeInfo(Initialization *const initialization){
    mesh_size_on_end_ = (*((*initialization).get_model_geometry())).get_width_of_end()
    /(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end();
    mesh_size_on_gap_ = (*((*initialization).get_model_geometry())).get_width_of_gap()
    /(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap();
    mesh_size_on_heater_ = (*((*initialization).get_model_geometry())).get_width_of_heater()
    /(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater();
    mesh_size_along_isolater_thickness_ = (*((*initialization).get_model_geometry())).get_thickness_of_isolater()
    /(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness();
    mesh_size_along_silicondioxide_thickness_ = (*((*initialization).get_model_geometry())).get_thickness_of_silicondioxide()
    /(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_silicondioxide_thickness();
    mesh_size_along_copper_thickness_=(*((*initialization).get_model_geometry())).get_thickness_of_copper()
    /(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_copper_thickness();
}

void GenerateMesh::CalculateCoordinates(Initialization *const initialization){
    int num_of_elements_between_ends=Constants::kNumOfHeaters_*((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()
                                                                +(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater());
    int new_ordering_after_left_end;
    int check_modulues;
    int mesh_seeds_on_end;
    int num_of_nodes=(*((*initialization).get_mesh_parameters())).get_num_of_nodes();
    int num_of_elements=(*((*initialization).get_mesh_parameters())).get_num_of_elements();
    
    x_coordinates_.clear();
    x_coordinates_.resize(num_of_nodes,0.0);
    x_coordinates_candidates.clear();
    x_coordinates_candidates.resize((*((*initialization).get_mesh_parameters())).get_dimensions_of_x(),0.0);
    y_coordinates_.clear();
    y_coordinates_.resize(num_of_nodes,0.0);
    y_coordinates_candidates.clear();
    y_coordinates_candidates.resize((*((*initialization).get_mesh_parameters())).get_dimensions_of_y(),0.0);
    
    for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_dimensions_of_x(); i++){
        
        if(i<=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end()){
            x_coordinates_candidates[i]=(mesh_size_on_end_*i);
        }
        
        else if(i>(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end() &&
                i<=((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end()
                    +(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2)){
                    x_coordinates_candidates[i] = x_coordinates_candidates[i-1] + mesh_size_on_gap_;
                }
        
        else if(i>((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end()
                   + (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2) &&
                i<=((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end() + num_of_elements_between_ends
                    -(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2)){
                    new_ordering_after_left_end = i-(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end()
                    -(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2-1;
                    check_modulues = new_ordering_after_left_end%((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater()
                                                                  +(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap());
                    
                    if(check_modulues<(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater()){
                        x_coordinates_candidates[i] = x_coordinates_candidates[i-1]+mesh_size_on_heater_;
                    }
                    if(check_modulues>=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater()){
                        x_coordinates_candidates[i] = x_coordinates_candidates[i-1]+mesh_size_on_gap_;
                    }
                }
        
        else if(i>((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end() + num_of_elements_between_ends
                   -(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2) &&
                i<=((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end() + num_of_elements_between_ends)){
            x_coordinates_candidates[i] = x_coordinates_candidates[i-1] + mesh_size_on_gap_;
        }
        
        else{
            x_coordinates_candidates[i] = x_coordinates_candidates[i-1] + mesh_size_on_end_;
        }
    }
    
    y_coordinates_candidates[0] = 0.0;
    y_coordinates_candidates[1] = 0.2;
    y_coordinates_candidates[2] = 0.4;
    y_coordinates_candidates[3] = 0.49;
    y_coordinates_candidates[Constants::kMeshSeedsAlongSiliconThickness_-1] = 0.499;
    y_coordinates_candidates[Constants::kMeshSeedsAlongSiliconThickness_]=(*((*initialization).get_model_geometry())).get_thickness_of_csilicon();
    
    int mesh_seeds_along_isolater_thickness=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness();
    for(int i=1; i<=mesh_seeds_along_isolater_thickness; i++){
        y_coordinates_candidates[Constants::kMeshSeedsAlongSiliconThickness_+i]=(*((*initialization).get_model_geometry())).get_thickness_of_csilicon()
        + i*mesh_size_along_isolater_thickness_;
    }
    
    y_coordinates_candidates[Constants::kMeshSeedsAlongSiliconThickness_+mesh_seeds_along_isolater_thickness+1]=(*((*initialization).get_model_geometry())).get_thickness_of_csilicon()
    +(*((*initialization).get_model_geometry())).get_thickness_of_isolater()+(*((*initialization).get_model_geometry())).get_thickness_of_titanium();
    
    for(int i=1; i<=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_silicondioxide_thickness(); i++){
        y_coordinates_candidates[Constants::kMeshSeedsAlongSiliconThickness_+mesh_seeds_along_isolater_thickness+1+i]=
        (*((*initialization).get_model_geometry())).get_thickness_of_csilicon()
        +(*((*initialization).get_model_geometry())).get_thickness_of_isolater()
        +(*((*initialization).get_model_geometry())).get_thickness_of_titanium()
        + i*mesh_size_along_silicondioxide_thickness_;
    }
    
    for(int i=1; i<=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_copper_thickness(); i++){
        y_coordinates_candidates[Constants::kMeshSeedsAlongSiliconThickness_+mesh_seeds_along_isolater_thickness+1
                                 +(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_silicondioxide_thickness()+i]
        = (*((*initialization).get_model_geometry())).get_thickness_of_csilicon()
        + (*((*initialization).get_model_geometry())).get_thickness_of_isolater()
        + (*((*initialization).get_model_geometry())).get_thickness_of_titanium()
        + (*((*initialization).get_model_geometry())).get_thickness_of_silicondioxide() + i*mesh_size_along_copper_thickness_;
    }
    
    //generate coordinates for nodes
    for(int j=0; j<(*((*initialization).get_mesh_parameters())).get_dimensions_of_y(); j++){
        for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_dimensions_of_x(); i++){
            x_coordinates_[(*((*initialization).get_mesh_parameters())).get_dimensions_of_x()*j+i]=x_coordinates_candidates[i];
            y_coordinates_[(*((*initialization).get_mesh_parameters())).get_dimensions_of_x()*j+i]=y_coordinates_candidates[j];
        }
    }
    //  printf("Generating mesh information completed\n");
}

void GenerateMesh::PrintCoordinatesResults(Initialization *const initialization){
    for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_num_of_nodes(); i++)
    printf("x_coordinates_[%d] = %f  y_coordinates_[%d] = %f\n", i, x_coordinates_[i], i, y_coordinates_[i]);
}


class DegreeOfFreedomAndEquationNumbers{
public:
    void InitializeDegreeOfFreedomAndEquationNumbers(Initialization *const);
    std::vector<int> &get_nodes_in_elements()
    {return nodes_in_elements_;}
    std::vector<int> &get_equation_numbers_of_nodes()
    {return equation_numbers_of_nodes_;}
    std::vector<int> &get_equation_numbers_in_elements()
    {return equation_numbers_in_elements_;}
    std::vector<int> &get_essential_bc_nodes()
    {return essential_bc_nodes_;}
    double get_num_of_essential_bc_nodes() const
    {return num_of_essential_bc_nodes_;}
    int get_num_of_equations() const
    {return num_of_equations_;}
    void set_essential_bc_nodes(Initialization* const, std::vector<double>&);
    void GenerateNodeAndEquationNumbersInElements(Initialization*);
    void PrintDofAndEquationNumbers(Initialization *const);
    
private:
    std::vector<int> nodes_in_elements_;
    std::vector<int> essential_bc_nodes_;
    int num_of_essential_bc_nodes_;
    std::vector<int> equation_numbers_of_nodes_;
    std::vector<int> equation_numbers_in_elements_;
    int num_of_equations_;
};
void DegreeOfFreedomAndEquationNumbers::InitializeDegreeOfFreedomAndEquationNumbers(Initialization *const initialization){
    num_of_essential_bc_nodes_=(*((*initialization).get_mesh_parameters())).get_dimensions_of_x();
}

void DegreeOfFreedomAndEquationNumbers::set_essential_bc_nodes(Initialization *const initialization, std::vector<double> &y_coordinates){
    essential_bc_nodes_.clear();
    essential_bc_nodes_.resize(num_of_essential_bc_nodes_, 0);
    int count_current_node_number=0;
    for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_num_of_nodes(); i++){
        if(y_coordinates[i]==Constants::kMinYCoordinate_){
            if(count_current_node_number>=num_of_essential_bc_nodes_){
                printf("number of nodes of essential boundary condition exceeds the right value\n");
                exit(-1);
            }
            essential_bc_nodes_[count_current_node_number]=i;
            ++count_current_node_number;
        }
    }
}

void DegreeOfFreedomAndEquationNumbers::GenerateNodeAndEquationNumbersInElements(Initialization *const initialization){
    //generate node numbering for each element
    int num_of_elements=(*((*initialization).get_mesh_parameters())).get_num_of_elements();
    int num_of_nodes=(*((*initialization).get_mesh_parameters())).get_num_of_nodes();
    
    nodes_in_elements_.clear();
    nodes_in_elements_.resize(Constants::kNumOfNodesInElement_*num_of_elements,0);
    for(int j=0; j<((*((*initialization).get_mesh_parameters())).get_dimensions_of_y()-1); j++){
        for(int i=0; i<((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1); i++){
            int temporary_node_id[4]={0,0,0,0};
            int element_number = j*((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1) + i;
            temporary_node_id[0] = ((j*(*((*initialization).get_mesh_parameters())).get_dimensions_of_x())+i);
            temporary_node_id[1] = temporary_node_id[0]+1;
            temporary_node_id[2] = (((j+1)*(*((*initialization).get_mesh_parameters())).get_dimensions_of_x()) + i + 1);
            temporary_node_id[3] = temporary_node_id[2]-1;
            nodes_in_elements_[0+element_number*Constants::kNumOfNodesInElement_] = temporary_node_id[0];
            nodes_in_elements_[1+element_number*Constants::kNumOfNodesInElement_] = temporary_node_id[1];
            nodes_in_elements_[2+element_number*Constants::kNumOfNodesInElement_] = temporary_node_id[2];
            nodes_in_elements_[3+element_number*Constants::kNumOfNodesInElement_] = temporary_node_id[3];
        }
    }
    
    equation_numbers_of_nodes_.clear();
    equation_numbers_of_nodes_.resize(num_of_nodes,0);
    int count_equation_number=0;
    for(int i=0; i<num_of_nodes; i++){
        int count_node=0;
        bool check_fixed_dof = false;
        while(count_node<num_of_essential_bc_nodes_){
            if(i==essential_bc_nodes_[count_node]){
                check_fixed_dof = true;
                break; //find the DOF K that has not been fixed
            }
            count_node++;
        }
        if(check_fixed_dof == false){
            equation_numbers_of_nodes_[i]=count_equation_number;
            count_equation_number++;
        }
        else if(check_fixed_dof == true){
            equation_numbers_of_nodes_[i]=-1;
        }
    }
    
    equation_numbers_in_elements_.clear();
    equation_numbers_in_elements_.resize(Constants::kNumOfNodesInElement_*num_of_elements,0);
    for(int i=0; i<num_of_elements; i++){
        for(int j=0; j<Constants::kNumOfNodesInElement_; j++){
            int current_node_number=nodes_in_elements_[j+i*Constants::kNumOfNodesInElement_];
            equation_numbers_in_elements_[j+i*Constants::kNumOfNodesInElement_]=equation_numbers_of_nodes_[current_node_number];
        }
    }
    
    num_of_equations_=equation_numbers_of_nodes_.back()+1;
}

void DegreeOfFreedomAndEquationNumbers::PrintDofAndEquationNumbers(Initialization *const initialization){
    for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_num_of_nodes(); i++)
    printf("equation_numbers_of_nodes_[%d]=%d\n", i, equation_numbers_of_nodes_[i]);
    printf("\n\nequation_numbers_in_elements_\n");
    for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_num_of_elements(); i++){
        for(int j=0; j<Constants::kNumOfNodesInElement_; j++)
        printf("%d  ", equation_numbers_in_elements_[j+i*Constants::kNumOfNodesInElement_]);
        printf("\n");
    }
}


class EachRowNozeroCount{
public:
    void SetDnnzAndOnnz(Initialization *, DegreeOfFreedomAndEquationNumbers * , int, int, PetscInt*, PetscInt*);
};
void EachRowNozeroCount::SetDnnzAndOnnz(Initialization *const initialization, DegreeOfFreedomAndEquationNumbers *const dof_and_equation_numbers, int equation_local_start, int equation_local_end_plus_one, PetscInt d_nnz[], PetscInt o_nnz[]){
    int num_of_equations = (*dof_and_equation_numbers).get_num_of_equations();
    int num_of_elements = (*((*initialization).get_mesh_parameters())).get_num_of_elements();
    std::vector<int>& equation_numbers_of_nodes=(*dof_and_equation_numbers).get_equation_numbers_of_nodes();
    std::vector<int> equation_numbers_in_elements=(*dof_and_equation_numbers).get_equation_numbers_in_elements();
    int local_row_number = 0;
    int diagonal_block_size = equation_local_end_plus_one - equation_local_start;
    int upper_bound_include = equation_local_end_plus_one - 1;
    int lower_bound_include = equation_local_start;
    std::list<int> neighbor_equation_list;
    
    for(int local_row_number = 0; local_row_number < diagonal_block_size; local_row_number++){
        d_nnz[local_row_number] = (PetscInt)1; // must at least have one (diagonally)
        o_nnz[local_row_number] = (PetscInt)0;

        int row_number = equation_local_start + local_row_number;
        
        for(int j=0;j<num_of_elements;j++){//searching...
            for(int k=0;k<Constants::kNumOfNodesInElement_;k++){ //searching...
                if(equation_numbers_in_elements[k+Constants::kNumOfNodesInElement_*j] == row_number){ //j-th element contains row_number
                
                    for(int l=0;l<Constants::kNumOfNodesInElement_;l++){
                        
                        int neighbor_equation_is = equation_numbers_in_elements[l+Constants::kNumOfNodesInElement_*j];
                        if (neighbor_equation_is >= 0){
                            //search if neighbor_equation_is is already in the list.
                            bool equation_already_in_list = false;
                            for (std::list<int>::iterator it = neighbor_equation_list.begin(); it != neighbor_equation_list.end(); ++it){
                                if(*it == neighbor_equation_is){
                                    equation_already_in_list = true;
                                    break;
                                }
                            }
                            if(equation_already_in_list == false){
                                neighbor_equation_list.push_back(neighbor_equation_is);
                            }
                        }
                    }
                    
                }// already found the elment
            }
        }

        //  classify those in the diagonal block and those in the off-diagonal block
        for (std::list<int>::iterator it = neighbor_equation_list.begin(); it != neighbor_equation_list.end(); ++it){
            if( *it <= upper_bound_include && *it >= lower_bound_include) {
                if(*it != row_number) {// must not be itself
                    d_nnz[local_row_number] += (PetscInt)1;
                }
            } else {
                o_nnz[local_row_number] += (PetscInt)1;
            }
        }

    }

}


// find boundary nodes
class BoundaryCondition{
public:
    void InitializeBoundaryCondition(Initialization *const);
    void FixTemperature(int, std::vector<std::vector<double> >&, std::vector<int>&, PETSC_STRUCT*);
    void PrintBoundaryConditionNodes(std::vector<int>& essential_bc_nodes){
        int num_of_essential_bc_nodes = essential_bc_nodes.size();
        for(int i=0;i<num_of_essential_bc_nodes;i++)
        printf("essential_bc_nodes_[%d] = %d\n", i, essential_bc_nodes[i]);
        printf("boundary_condition_temperature_ = %f\n", boundary_condition_temperature_);
    }
    
private:
    double boundary_condition_temperature_;
};
void BoundaryCondition::InitializeBoundaryCondition(Initialization *const initialization){
    boundary_condition_temperature_=(*((*initialization).get_analysis_constants())).get_boundary_condition_temperature();
}

void BoundaryCondition::FixTemperature(const int element_number, std::vector<std::vector<double> >&element_stiffness_matrix, std::vector<int>&equation_numbers_in_elements, PETSC_STRUCT* obj){
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        int row_equation_number=equation_numbers_in_elements[k+Constants::kNumOfNodesInElement_*element_number];
        for(int jj=0;jj<Constants::kNumOfNodesInElement_;jj++){//loop over all force components in this element
            int column_equation_number=equation_numbers_in_elements[jj+Constants::kNumOfNodesInElement_*element_number];
            if(row_equation_number>=0 && column_equation_number<0){
                // this item goes directly to rhs (ADD UP)
                PetscErrorCode ierr = VecSetValue(obj->rhs, (PetscInt)row_equation_number, (PetscScalar)(-(element_stiffness_matrix[k][jj]*boundary_condition_temperature_)), ADD_VALUES);
//                (obj->heat_load)[row_equation_number] += -(element_stiffness_matrix[k][jj]*boundary_condition_temperature_);
            }
        }//for
    }//for
    //  printf("Processing fixed temperature boundary condition completed\n");
}


class MappingShapeFunctionAndDerivatives{
public:
    void InitializeMappingShapeFunctionAndDerivatives();
    void set_coordinates_in_this_element(int, std::vector<int>&, std::vector<double>&, std::vector<double>&);
    void set_shape_function(double, double);
    void set_shape_function_derivatives(double, double);
    void set_determinant_of_jacobian_matrix();
    void set_dn_dx();
    void PrintDeterminantOfJacobianMatrix();
    
protected:
    std::vector<std::vector<double> > coordinates_in_this_element_;
    std::vector<double> shape_function_;
    std::vector<std::vector<double> > shape_function_derivatives_;
    std::vector<std::vector<double> > dn_dx_;
    double determinant_of_jacobian_matrix_;
    double jacobian_matrix_[2][2];
};
void MappingShapeFunctionAndDerivatives::InitializeMappingShapeFunctionAndDerivatives(){
    coordinates_in_this_element_.resize(2);
    coordinates_in_this_element_[0].resize(Constants::kNumOfNodesInElement_, 0.0);
    coordinates_in_this_element_[1].resize(Constants::kNumOfNodesInElement_, 0.0);
    
    shape_function_.resize(Constants::kNumOfNodesInElement_, 0.0);
    
    shape_function_derivatives_.resize(2);
    shape_function_derivatives_[0].resize(Constants::kNumOfNodesInElement_, 0.0);
    shape_function_derivatives_[1].resize(Constants::kNumOfNodesInElement_, 0.0);
    
    dn_dx_.resize(2);
    dn_dx_[0].resize(Constants::kNumOfNodesInElement_, 0.0);
    dn_dx_[1].resize(Constants::kNumOfNodesInElement_, 0.0);
}

void MappingShapeFunctionAndDerivatives::set_coordinates_in_this_element(const int element_number, std::vector<int>& nodes_in_elements,
                                                                         std::vector<double>& x_coordinates, std::vector<double>& y_coordinates){
    for (int k=0;k<Constants::kNumOfNodesInElement_;k++){
        coordinates_in_this_element_[0][k]=x_coordinates[nodes_in_elements[k+element_number*Constants::kNumOfNodesInElement_]];
        coordinates_in_this_element_[1][k]=y_coordinates[nodes_in_elements[k+element_number*Constants::kNumOfNodesInElement_]];
    }
}

void MappingShapeFunctionAndDerivatives::set_shape_function(const double ksi_coordinate, const double eta_coordinate){
    shape_function_[0] = 0.25*(1-ksi_coordinate)*(1-eta_coordinate);
    shape_function_[1] = 0.25*(1+ksi_coordinate)*(1-eta_coordinate);
    shape_function_[2] = 0.25*(1+ksi_coordinate)*(1+eta_coordinate);
    shape_function_[3] = 0.25*(1-ksi_coordinate)*(1+eta_coordinate);
}

void MappingShapeFunctionAndDerivatives::set_shape_function_derivatives(const double ksi_coordinate, const double eta_coordinate){
    //----------------------to get jacobian determinant----------------------
    // Calculate the local derivatives of the shape functions.
    shape_function_derivatives_[0][0] = -0.25*(1 - eta_coordinate);
    shape_function_derivatives_[1][0] = -0.25*(1 - ksi_coordinate);
    shape_function_derivatives_[0][1] =  0.25*(1 - eta_coordinate);
    shape_function_derivatives_[1][1] = -0.25*(1 + ksi_coordinate);
    shape_function_derivatives_[0][2] =  0.25*(1 + eta_coordinate);
    shape_function_derivatives_[1][2] =  0.25*(1 + ksi_coordinate);
    shape_function_derivatives_[0][3] = -0.25*(1 + eta_coordinate);
    shape_function_derivatives_[1][3] =  0.25*(1 - ksi_coordinate);
}

void MappingShapeFunctionAndDerivatives::set_determinant_of_jacobian_matrix(){
    for(int u=0;u<2;u++){
        for(int v=0;v<2;v++){
            determinant_of_jacobian_matrix_=0.0;
            for(int m=0;m<Constants::kNumOfNodesInElement_;m++){
                determinant_of_jacobian_matrix_ += shape_function_derivatives_[u][m]*coordinates_in_this_element_[v][m];
            }
            jacobian_matrix_[u][v]=determinant_of_jacobian_matrix_;
        }
    }
    determinant_of_jacobian_matrix_=jacobian_matrix_[0][0]*jacobian_matrix_[1][1]-jacobian_matrix_[0][1]*jacobian_matrix_[1][0];
    determinant_of_jacobian_matrix_=fabs(determinant_of_jacobian_matrix_);
}

void MappingShapeFunctionAndDerivatives::set_dn_dx(){
    //-------------Jacobian inverse----------------------
    double reciprocal=1.0/determinant_of_jacobian_matrix_;
    double inverse_of_jacobian[2][2];
    inverse_of_jacobian[0][0]=reciprocal*jacobian_matrix_[1][1];
    inverse_of_jacobian[1][1]=reciprocal*jacobian_matrix_[0][0];
    inverse_of_jacobian[0][1]=-reciprocal*jacobian_matrix_[0][1];
    inverse_of_jacobian[1][0]=-reciprocal*jacobian_matrix_[1][0];
    //------------Derivatives of shape function w.r.t. global coordinate--------------
    for(int u=0;u<Constants::kNumOfNodesInElement_;u++){
        for(int v=0;v<2;v++){
            dn_dx_[v][u]=0.0;
            for(int m=0;m<2;m++){
                dn_dx_[v][u] += inverse_of_jacobian[m][v]*shape_function_derivatives_[m][u];
            }
        }
    }
}

void MappingShapeFunctionAndDerivatives::PrintDeterminantOfJacobianMatrix(){
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("coordinates_in_this_element_[0][%d] = %f\n", k, coordinates_in_this_element_[0][k]);
        printf("coordinates_in_this_element_[1][%d] = %f\n", k, coordinates_in_this_element_[1][k]);
    }
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("shape_function_[%d] = %f\n", k, shape_function_[k]);
    }
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("shape_function_derivatives_[0][%d] = %f\n", k, shape_function_derivatives_[0][k]);
        printf("shape_function_derivatives_[1][%d] = %f\n", k, shape_function_derivatives_[1][k]);
    }
    printf("determinant_of_jacobian_matrix_ = %f\n", determinant_of_jacobian_matrix_);
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("dn_dx_[0][%d] = %f\n", k, dn_dx_[0][k]);
        printf("dn_dx_[1][%d] = %f\n", k, dn_dx_[1][k]);
    }
}


class TemperatureDependentVariables{
public:
    void InitializeTemperatureDependentVariables(Initialization *const);
    double get_thermal_conductivity(int, double, std::vector<int>&);
    double get_thermal_conductivity_derivative(int, double, std::vector<int>&);
    double get_body_heat_flux(double, double);
    double get_body_heat_flux_derivative(double, double);
    double get_heater_crosssection_area() const
    {return heater_crosssection_area_mm_square_;}
    double get_specific_heat(int, double, std::vector<int>&);
    double get_emissivity(double);
//    double get_emissivity_derivative(double);
    double get_heater_crosssection_area_mm_square() const
    {return heater_crosssection_area_mm_square_;}
//    void PrintTemperatureDependentVariables(std::vector<int>&,Initialization *const);
    
private:
    double heater_crosssection_area_mm_square_;
};
void TemperatureDependentVariables::InitializeTemperatureDependentVariables(Initialization *const initialization){
    heater_crosssection_area_mm_square_=(*((*initialization).get_model_geometry())).get_thickness_of_titanium()
    *(*((*initialization).get_model_geometry())).get_width_of_heater();
}

double TemperatureDependentVariables::get_thermal_conductivity(const int element_number, const double temperature, std::vector<int>& material_id_of_elements){
    double conductivity;
    int material_id=material_id_of_elements[element_number];
    switch(material_id){
        case 0: conductivity=0.0002723*temperature*temperature-0.5435*temperature+295.9; break; //csilicon
        case 1: conductivity=1.375e-05*temperature*temperature-0.01653*temperature+22.72; break;  //titanium
        case 2: conductivity=4.8e-05*temperature*temperature-0.06094*temperature+24.24; break; //silicondioxide
        case 3: conductivity=-9.085e-06*temperature*temperature-0.05699*temperature+405.6987; break; // copper
    }
    return conductivity;
}

double TemperatureDependentVariables::get_thermal_conductivity_derivative(const int element_number, const double temperature, std::vector<int>& material_id_of_elements){
    double conductivity_derivative;
    int material_id=material_id_of_elements[element_number];
    switch(material_id){
        case 0: conductivity_derivative=0.0002723*2*temperature-0.5435; break;
        case 1: conductivity_derivative=1.375e-05*2*temperature-0.01653; break;
        case 2: conductivity_derivative=4.8e-05*2*temperature-0.06094; break;
        case 3: conductivity_derivative=-9.085e-06*2*temperature-0.05699; break;
    }
    return conductivity_derivative;
}

double TemperatureDependentVariables::get_specific_heat(const int element_number, const double temperature, std::vector<int>& material_id_of_elements){ //Cal/g/K = 4.184e9 mJ/tonne/K
    double specific_heat;
    int material_id=material_id_of_elements[element_number];
    switch(material_id){
        case 0: specific_heat=-632.6*temperature*temperature+9.952e+5*temperature+5.189e+8; break;//csilicon
        case 1: specific_heat=714.2*temperature*temperature-6.233e+5*temperature+7.128e+8; break;//titanium
        case 2: specific_heat=-437.4*temperature*temperature+1.404e+6*temperature+3.903e+8; break; //silicondioxide
        case 3: specific_heat=180.2*temperature*temperature-1.723e+5*temperature+4.175e+8; break;// copper
    }
    return specific_heat;
}

double TemperatureDependentVariables::get_emissivity(const double temperature){ //Cal/g/K = 4.184e9 mJ/tonne/K
    //data from curve 52 P148
    double emissivity;
    emissivity=-1.932e-7*temperature*temperature+0.0003696*temperature+0.07681;// copper
    return emissivity;
}
/*
double TemperatureDependentVariables::get_emissivity_derivative(const double temperature){ //Cal/g/K = 4.184e9 mJ/tonne/K
    //data from curve 52 P148
    double emissivity_derivative;
    emissivity_derivative=-1.932e-7*2*temperature+0.0003696;// copper
    return emissivity_derivative;
}
*/

double TemperatureDependentVariables::get_body_heat_flux(const double temperature, const double current_in_element_ma){ //Cal/g/K = 4.184e9 mJ/tonne/K
    //data from
    //Bel’skaya, E. A. "An experimental investigation of the electrical resistivity of titanium in the temperature range from 77 to 1600 K." High Temperature 43.4 (2005): 546-553.
    //---------------------
    double body_heat_flux;
    double resistivity;
    double current_in_element_a = current_in_element_ma*1.0e-3;  // mA -> A
    double heater_crosssection_area_m_square=heater_crosssection_area_mm_square_*1.0e-3*1.0e-3;  //mm^2->m^2
//    resistivity = 1.403e-15*temperature*temperature + 1.037e-8*temperature - 2.831e-6;// titanium heater
    resistivity = -6.674e-13*temperature*temperature + 2.462e-9*temperature - 1.893e-7;// resistivity used in the IPISE paper.

    body_heat_flux = resistivity*pow((current_in_element_a/heater_crosssection_area_m_square),2);
    body_heat_flux *= 1.0e-6;
    return body_heat_flux;
}

double TemperatureDependentVariables::get_body_heat_flux_derivative(const double temperature, const double current_in_element_ma){ //Cal/g/K = 4.184e9 mJ/tonne/K
    //data from
    //Bel’skaya, E. A. "An experimental investigation of the electrical resistivity of titanium in the temperature range from 77 to 1600 K." High Temperature 43.4 (2005): 546-553.
    //---------------------
    double resistivity_derivative;
    double body_heat_flux_derivative;
    double current_in_element_a=current_in_element_ma*1.0e-3;
    double heater_crosssection_area_m_square=heater_crosssection_area_mm_square_*1.0e-3*1.0e-3;  //mm^2->m^2
    resistivity_derivative=1.403e-15*2*temperature+1.037e-8;// titanium heater
    body_heat_flux_derivative = resistivity_derivative*pow((current_in_element_a/heater_crosssection_area_m_square),2);
    body_heat_flux_derivative *= 1.0e-6;
    return body_heat_flux_derivative;
}
/*
void TemperatureDependentVariables::PrintTemperatureDependentVariables(std::vector<int>& material_id_of_elements, Initialization *const initialization){
    for(int i=0;i<Constants::kNumOfHeaters_;i++){
        printf("currents_in_heater_[%d] = %f\n", i, (*((*initialization).get_currents_in_heater())).get_current_in_heater()[i]);
    }
    printf("get_heater_crosssection_area = %f\n", heater_crosssection_area_mm_square_);
    printf("get_thermal_conductivity = %f\n", get_thermal_conductivity(1, 300.0, material_id_of_elements));
    printf("get_thermal_conductivity_derivative = %f\n", get_thermal_conductivity_derivative(1, 300.0, material_id_of_elements));
    printf("get_specific_heat = %f\n", get_specific_heat(1, 300.0, material_id_of_elements));
    printf("get_emissivity = %f\n", get_emissivity(300.0));
    printf("get_emissivity_derivative = %f\n", get_emissivity_derivative(300.0));
    printf("get_body_heat_flux = %f\n", get_body_heat_flux(300.0, 50.0));
    printf("get_body_heat_flux_derivative = %.16f\n", get_body_heat_flux_derivative(300.0, 50.0));
}
*/

class HeaterElements:public MappingShapeFunctionAndDerivatives {
public:
    void InitializeHeaterElements(Initialization *const);
    void set_elements_as_heater(Initialization *const);
    double get_num_of_elements_as_heater() const
    {return num_of_elements_as_heater_;}
    std::vector<int> &get_elements_as_heater()
    {return elements_as_heater_;}
    void HeatSupply(int, int, PETSC_STRUCT*, std::vector<int>&, std::vector<int>&, std::vector<double>&, TemperatureDependentVariables*, Initialization *const);
    void PrintHeaterElements(){
        for(int i=0;i<num_of_elements_as_heater_;i++)
        printf("elements_as_heater_[%d] = %d\n", i, elements_as_heater_[i]);
    }
    
private:
    int num_of_elements_as_heater_;
    std::vector<int> elements_as_heater_;
};
void HeaterElements::InitializeHeaterElements(Initialization *const initialization){
    num_of_elements_as_heater_ = (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater()*Constants::kNumOfHeaters_;
    elements_as_heater_.resize(num_of_elements_as_heater_, 0);
    InitializeMappingShapeFunctionAndDerivatives();
}

void HeaterElements::set_elements_as_heater(Initialization *const initialization){
    int dimensions_of_x = (*((*initialization).get_mesh_parameters())).get_dimensions_of_x();
    int mesh_seeds_on_end = (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end();
    int mesh_seeds_on_gap = (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap();
    int mesh_seeds_on_heater = (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater();
    int first_heater_element_=(Constants::kMeshSeedsAlongSiliconThickness_+(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness())
    *(dimensions_of_x-1)+mesh_seeds_on_end+mesh_seeds_on_gap/2;
    for(int i=0; i<Constants::kNumOfHeaters_; i++){
        for(int j=0; j<mesh_seeds_on_heater; j++){
            elements_as_heater_[i*mesh_seeds_on_heater+j] = first_heater_element_ + i*(mesh_seeds_on_gap+mesh_seeds_on_heater) + j;
        }
    }
}

void HeaterElements::HeatSupply(const int element_number, const int heater_element_number, PETSC_STRUCT* obj, std::vector<int>&nodes_in_elements, std::vector<int>&equation_numbers_in_elements, std::vector<double>&current_temperature_field, TemperatureDependentVariables *const temperature_dependent_variables, Initialization *const initialization){//Calculate internal load contribution
    //integration rule
    int num_of_integration_points=2;
    double coordinates_of_integration_points[2]={-0.57735026, 0.57735026}; //gaussian quadrature coordinates
    double weights_of_integration_points[2]={1.0, 1.0};//weight of gaussian point
    
    int heater_number=heater_element_number/(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater();
    double current=(*((*initialization).get_currents_in_heater())).get_current_in_heater()[heater_number];
    
    for(int k=0;k<num_of_integration_points;k++){
        double ksi_coordinate=coordinates_of_integration_points[k];  //gaussian piont coordinate
        double ksi_weight=weights_of_integration_points[k];    //weight of gaussian quadrature
        for(int l=0;l<num_of_integration_points;l++){
            double eta_coordinate=coordinates_of_integration_points[l];
            double eta_weight=weights_of_integration_points[l];
            
            set_shape_function(ksi_coordinate, eta_coordinate);
            set_shape_function_derivatives(ksi_coordinate, eta_coordinate);
            set_determinant_of_jacobian_matrix();
            
            double temperature=0.0;
            for(int ii=0;ii<Constants::kNumOfNodesInElement_;ii++)
            temperature += current_temperature_field[(nodes_in_elements[ii+element_number*4])]*shape_function_[ii];
            
            for(int j=0;j<Constants::kNumOfNodesInElement_;j++){
                double equation_number=equation_numbers_in_elements[j+element_number*4];
                if(equation_number>=0){
                    PetscScalar add_term = shape_function_[j]*(*temperature_dependent_variables).get_body_heat_flux(temperature,current)*determinant_of_jacobian_matrix_*ksi_weight*eta_weight;
                    PetscErrorCode ierr = VecSetValue(obj->rhs, (PetscInt)equation_number, add_term, ADD_VALUES);
/*
                    heat_load[equation_number] += shape_function_[j]*
                    (*temperature_dependent_variables).get_body_heat_flux(temperature,current)*determinant_of_jacobian_matrix_*ksi_weight*eta_weight;
 */
                }//if
            }//for j
        }//for l
    }//for k
    //  printf("processing heat supply element completed\n");
}

// find radiation elements
class RadiationElements{
public:
    void InitializeRadiationElements(Initialization *const);
    void set_elements_with_radiation(Initialization *const);
    double get_num_of_elements_with_radiation() const
    {return num_of_elements_with_radiation_;}
    std::vector<int> &get_elements_with_radiation()
    {return elements_with_radiation_;}
    void PrintRadiationElements();
    
private:
    int num_of_elements_with_radiation_;
    std::vector<int> elements_with_radiation_;
};
void RadiationElements::InitializeRadiationElements(Initialization *const initialization){
    num_of_elements_with_radiation_=(*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1;
}

void RadiationElements::set_elements_with_radiation(Initialization *const initialization){
    elements_with_radiation_.resize(num_of_elements_with_radiation_, 0);
    for(int i=0; i<num_of_elements_with_radiation_; i++){
        elements_with_radiation_[i] = ((*((*initialization).get_mesh_parameters())).get_dimensions_of_y()-2)
        *num_of_elements_with_radiation_+ i;
    }
}

void RadiationElements::PrintRadiationElements(){
    for(int i=0;i<num_of_elements_with_radiation_;i++){
        printf("elements_with_radiation_[%d] = %d\n", i, elements_with_radiation_[i]);
    }
}


class TemperatureFieldInitial{
public:
    void set_initial_temperature_field(std::vector<int>&, std::vector<double>&, PETSC_STRUCT*, Initialization *const, std::vector<int>&);
//    void PrintInitialTemperatureField(std::vector<double>&);
};
void TemperatureFieldInitial::set_initial_temperature_field(std::vector<int>&essential_bc_nodes, std::vector<double>&current_temperature_field, PETSC_STRUCT* obj, Initialization *const initialization, std::vector<int>& equation_numbers_of_nodes){
    double boundary_condition_temperature_=(*((*initialization).get_analysis_constants())).get_boundary_condition_temperature();
    double sample_initial_temperature_=(*((*initialization).get_analysis_constants())).get_sample_initial_temperature();
    int num_of_nodes = (*((*initialization).get_mesh_parameters())).get_num_of_nodes();
    for(int i=0; i<num_of_nodes; i++){
        if(equation_numbers_of_nodes[i]<0) {
            current_temperature_field[i] = boundary_condition_temperature_;
        } else {
            current_temperature_field[i] = sample_initial_temperature_;
            PetscErrorCode ierr = VecSetValue(obj->current_temperature_field_local, (PetscInt)(equation_numbers_of_nodes[i]), (PetscScalar)sample_initial_temperature_, ADD_VALUES);
        }
    }
}

/*
void TemperatureFieldInitial::PrintInitialTemperatureField(std::vector<double>&initial_temperature_field){
    for(int i=0;i<initial_temperature_field.size();i++){
        printf("initial_temperature_field[%d] = %f\n", i, initial_temperature_field[i]);
    }
}
*/

class MaterialParameters{
public:
    void set_material_id_of_elements(Initialization *const);
    void set_densities();
    std::vector<int>& get_material_id_of_elements(){return material_id_of_elements_;}
    std::vector<double>& get_densities(){return densities_;}
    void PrintMaterialParameters();
    
private:
    std::vector<double> densities_;
    std::vector<int> material_id_of_elements_;
};
void MaterialParameters::set_densities(){
    densities_.resize(Constants::kNumOfMaterials_,0.0);
    //material information density [tonne]/[mm]^3
    densities_[0] = 2.329e-9;  // csilicon
    densities_[1] = 4.506e-9;  // titanium (heater)
    densities_[2] = 2.65e-9; // silicon dioxide
    densities_[3] = 8.96e-9; // copper
}

void MaterialParameters::set_material_id_of_elements(Initialization *const initialization){
    material_id_of_elements_.resize((*((*initialization).get_mesh_parameters())).get_num_of_elements(),0);
    int first_element_of_silicon = 0;
    int last_element_of_silicon = Constants::kMeshSeedsAlongSiliconThickness_
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1)-1;
    //isolater film
    int first_element_of_isolater=Constants::kMeshSeedsAlongSiliconThickness_
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1);
    int last_element_of_isolater=(Constants::kMeshSeedsAlongSiliconThickness_+(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness())
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1)-1;
    
    //heater and silicon dioxide mix layer (ends excluded!!!)
    int first_element_of_heater_gap_mixed = (Constants::kMeshSeedsAlongSiliconThickness_+(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness())
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1)
    + (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end();
    int last_element_of_heater_gap_mixed = (Constants::kMeshSeedsAlongSiliconThickness_+(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness()+1)
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1)-1
    - (*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_end();
    int first_element_of_pure_layer_of_silicondioxide = (Constants::kMeshSeedsAlongSiliconThickness_+(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness()+1)
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1);
    int last_element_of_pure_layer_of_silicondioxide = (Constants::kMeshSeedsAlongSiliconThickness_+(*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_isolater_thickness()+1
                                                        + (*((*initialization).get_mesh_parameters())).get_mesh_seeds_along_silicondioxide_thickness())
    *((*((*initialization).get_mesh_parameters())).get_dimensions_of_x()-1)-1;
    int first_element_of_copper = last_element_of_pure_layer_of_silicondioxide+1;
    int last_element_of_copper = (*((*initialization).get_mesh_parameters())).get_num_of_elements()-1;
    
    for(int i=0; i<(*((*initialization).get_mesh_parameters())).get_num_of_elements(); i++){
        //csilicon
        if(i<=last_element_of_silicon){
            material_id_of_elements_[i]=0;
        }
        
        // isolater layer
        else if(i>=first_element_of_isolater && i<=last_element_of_isolater){
            material_id_of_elements_[i]=2;
        }
        
        // left end on the mix layer
        else if(i>last_element_of_silicon && i<first_element_of_heater_gap_mixed){
            material_id_of_elements_[i]=2;
        }
        
        //heater and silicon dioxide mixed layer
        else if(i>=first_element_of_heater_gap_mixed && i<=last_element_of_heater_gap_mixed){
            if( i-first_element_of_heater_gap_mixed>=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2 &&
               last_element_of_heater_gap_mixed-i>=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2 ){
                
                int new_label_start_from_zero = i-first_element_of_heater_gap_mixed-(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()/2;
                int check_gap_heater_belonging = (new_label_start_from_zero%((*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_gap()
                                                                             +(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater()));
                if(check_gap_heater_belonging<(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater())
                material_id_of_elements_[i]=1;
                if(check_gap_heater_belonging>=(*((*initialization).get_mesh_parameters())).get_mesh_seeds_on_heater())
                material_id_of_elements_[i]=2;
            }
            else material_id_of_elements_[i]=2; //Elements belong to gap
        }
        
        // right end on the mix layer
        else if(i>last_element_of_heater_gap_mixed && i<first_element_of_pure_layer_of_silicondioxide){
            material_id_of_elements_[i]=2;
        }
        
        //silicon dioxide
        else if(i>=first_element_of_pure_layer_of_silicondioxide && i<=last_element_of_pure_layer_of_silicondioxide){
            material_id_of_elements_[i]=2;
        }
        
        //copper
        else if(i>=first_element_of_copper && i<=last_element_of_copper){
            material_id_of_elements_[i]=3;
        }
        
        else if(i>last_element_of_copper){
            printf("Asigning material id to elements is wrong\n");
            exit(-1);
        }
    }
}

void MaterialParameters::PrintMaterialParameters(){
    for(int i=0;i<Constants::kNumOfMaterials_;i++){
        printf("densities_[%d]= %f\n",i,densities_[i]);
    }
    for(int i=0;i<material_id_of_elements_.size();i++){
        printf("material_id_of_elements_[%d] = %d\n", i, material_id_of_elements_[i]);
    }
}


class IntegrationOverEdge{
public:
    void InitializeIntegrationOverEdge();
    void EdgeIntegration(int, double, std::vector<double>&,std::vector<int>&);
    void PrintDeterminantOfJacobianMatrix();
    
protected:
    std::vector<std::vector<double> > coordinates_in_this_element_;
    std::vector<double> shape_function_;
    std::vector<std::vector<double> > shape_function_derivatives_;
    double determinant_of_jacobian_matrix_;
};
void IntegrationOverEdge::InitializeIntegrationOverEdge(){
    coordinates_in_this_element_.resize(2);
    coordinates_in_this_element_[0].resize(Constants::kNumOfNodesInElement_, 0.0);
    coordinates_in_this_element_[1].resize(Constants::kNumOfNodesInElement_, 0.0);
    
    shape_function_.resize(Constants::kNumOfNodesInElement_, 0.0);
    
    shape_function_derivatives_.resize(2);
    shape_function_derivatives_[0].resize(Constants::kNumOfNodesInElement_, 0.0);
    shape_function_derivatives_[1].resize(Constants::kNumOfNodesInElement_, 0.0);
}

void IntegrationOverEdge::EdgeIntegration(const int element_number, const double ksi_coordinate, std::vector<double>& x_coordinates, std::vector<int>& nodes_in_elements){
    double Partial_X_Partial_ksi,Partial_Y_Partial_ksi;
    //zero vector and matrices
    int num_of_nodes_in_one_dimensional_element=2;
    for(int k=0;k<num_of_nodes_in_one_dimensional_element;k++){
        coordinates_in_this_element_[0][k]=0.0;
        coordinates_in_this_element_[1][k]=0.0;
        shape_function_[k]=0.0;
        shape_function_derivatives_[0][k]=0.0;
        shape_function_derivatives_[1][k]=0.0;
    }
    
    //order: right-->left
    coordinates_in_this_element_[0][0]=x_coordinates[nodes_in_elements[2+element_number*Constants::kNumOfNodesInElement_]];
    coordinates_in_this_element_[0][1]=x_coordinates[nodes_in_elements[3+element_number*Constants::kNumOfNodesInElement_]];
    
    shape_function_[0] = 0.5*(1+ksi_coordinate);
    shape_function_[1] = 0.5*(1-ksi_coordinate);
    
    //----------------------to get determinant_of_jacobian_matrix----------------------
    // Calculate the local derivatives of the shape functions.
    shape_function_derivatives_[0][0] = 0.5;
    shape_function_derivatives_[0][1] = -0.5;
    
    Partial_X_Partial_ksi = shape_function_derivatives_[0][0]*coordinates_in_this_element_[0][0] +
    shape_function_derivatives_[0][1]*coordinates_in_this_element_[0][1];
    
    determinant_of_jacobian_matrix_=fabs(Partial_X_Partial_ksi);
}

void IntegrationOverEdge::PrintDeterminantOfJacobianMatrix(){
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("coordinates_in_this_element_[0][%d] = %f\n", k, coordinates_in_this_element_[0][k]);
        printf("coordinates_in_this_element_[1][%d] = %f\n", k, coordinates_in_this_element_[1][k]);
    }
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("shape_function_[%d] = %f\n", k, shape_function_[k]);
    }
    for(int k=0;k<Constants::kNumOfNodesInElement_;k++){
        printf("shape_function_derivatives_[0][%d] = %f\n", k, shape_function_derivatives_[0][k]);
        printf("shape_function_derivatives_[1][%d] = %f\n", k, shape_function_derivatives_[1][k]);
    }
    printf("determinant_of_jacobian_matrix_ = %f\n", determinant_of_jacobian_matrix_);
}


class ElementalMatrix:public MappingShapeFunctionAndDerivatives{
public:
    void InitializeElementalMatrix();
    void set_element_stiffness_matrix(int, std::vector<int>&, std::vector<int>&, std::vector<double>&, TemperatureDependentVariables*);
    void MapElementalToGlobalStiffness(PETSC_STRUCT*, std::vector<int>&, int);
    std::vector<std::vector<double> >& get_element_stiffness_matrix(){return element_stiffness_matrix_;}
//    void PrintStiffnessMatrix(std::vector<double>&);

private:
    std::vector<std::vector<double> > element_stiffness_matrix_;
    std::vector<std::vector<double> > element_mass_matrix_;
};
void ElementalMatrix::InitializeElementalMatrix(){
    element_stiffness_matrix_.resize(Constants::kNumOfNodesInElement_);
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        element_stiffness_matrix_[i].resize(Constants::kNumOfNodesInElement_,0.0);
    }
    element_mass_matrix_.resize(Constants::kNumOfNodesInElement_);
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        element_mass_matrix_[i].resize(Constants::kNumOfNodesInElement_,0.0);
    }
    InitializeMappingShapeFunctionAndDerivatives();
}

void ElementalMatrix::set_element_stiffness_matrix(const int element_number, std::vector<int>&nodes_in_elements, std::vector<int>& material_id_of_elements, std::vector<double>& current_temperature_field, TemperatureDependentVariables *const temperature_dependent_variables){
    
    //----------------integration rule------------------
    int num_of_integration_points = 3;
    double coordinates_of_integration_points[3]={-0.7745966692, 0, 0.7745966692}; //gaussian quadrature coordinates
    double weights_of_integration_points[3]={0.5555555555, 0.8888888888, 0.5555555555};//weight of gaussian point
    
    //---------zero out element K   matrix--------------
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        for(int j=0;j<Constants::kNumOfNodesInElement_;j++){
            element_stiffness_matrix_[i][j]=0.0;
        }
    }
    
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        for(int j=0;j<Constants::kNumOfNodesInElement_;j++){
            
            for(int k=0;k<num_of_integration_points;k++){
                double ksi_coordinate = coordinates_of_integration_points[k];  //gaussian piont coordinate
                double ksi_weight = weights_of_integration_points[k];    //weight of gaussian quadrature
                
                for(int l=0;l<num_of_integration_points;l++){
                    double eta_coordinate = coordinates_of_integration_points[l];
                    double eta_weight = weights_of_integration_points[l];
                    
                    set_shape_function(ksi_coordinate, eta_coordinate);
                    set_shape_function_derivatives(ksi_coordinate, eta_coordinate);
                    set_determinant_of_jacobian_matrix();
                    set_dn_dx();
                    
                    double dni_x=dn_dx_[0][i];
                    double dni_y=dn_dx_[1][i];
                    double dnj_x=dn_dx_[0][j];
                    double dnj_y=dn_dx_[1][j];
                    
                    double nx_nx = dni_x*dnj_x*determinant_of_jacobian_matrix_*ksi_weight*eta_weight;
                    double ny_ny = dni_y*dnj_y*determinant_of_jacobian_matrix_*ksi_weight*eta_weight;
                    
                    double temperature = 0.0;
                    for(int ii=0;ii<Constants::kNumOfNodesInElement_;ii++){
                        temperature += current_temperature_field[(nodes_in_elements[ii+element_number*4])]*shape_function_[ii];
                    }
                    
                    element_stiffness_matrix_[i][j] +=
                    (*temperature_dependent_variables).get_thermal_conductivity(element_number, temperature, material_id_of_elements)*(nx_nx+ny_ny);
                }
            }
        }
    }
}

void ElementalMatrix::MapElementalToGlobalStiffness(PETSC_STRUCT* obj, std::vector<int>&equation_numbers_in_elements, const int element_number){
    PetscErrorCode ierr;
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        for(int j=0;j<Constants::kNumOfNodesInElement_;j++){
            int row_equation_number=equation_numbers_in_elements[i+element_number*Constants::kNumOfNodesInElement_];
            int column_equation_number=equation_numbers_in_elements[j+element_number*Constants::kNumOfNodesInElement_];
            if(row_equation_number>=0 && column_equation_number>=0){
            //            if(row_equation_number>=0 && column_equation_number>=0 && column_equation_number<=row_equation_number){
                // first add this item to stiffness_matrix for constructing the rhs later.
                // Note here, the term added up stiffness matrix is forced to take a negative sign. This is because we want to subtract stiffness_matrix(original) * current_temperature_field from rhs, but PETsc function MatMultAdd() can only do addition operation.
                #ifdef radiation
                ierr = MatSetValue(obj->stiffness_matrix,(PetscInt)row_equation_number,(PetscInt)column_equation_number,(PetscScalar)(-element_stiffness_matrix_[i][j]), ADD_VALUES);
                #endif
                //  also, we need to add this directly to the Amat.
                ierr = MatSetValue(obj->Amat,(PetscInt)row_equation_number,(PetscInt)column_equation_number,(PetscScalar)(element_stiffness_matrix_[i][j]), ADD_VALUES);
                /*
                int position_in_desparsed_matrix=accumulative_half_band_width_vector[row_equation_number]-(row_equation_number-column_equation_number);
                stiffness_matrix[position_in_desparsed_matrix] += element_stiffness_matrix_[i][j];
                */
            }
        }
    }
    //printf("map to global stiffness matrix completed\n");
}
/*
void ElementalStiffnessMatrix::PrintStiffnessMatrix(std::vector<double>&stiffness_matrix){
    int size_of_desparsed_stiffness_matrix = stiffness_matrix.size();
    for(int i=0; i<size_of_desparsed_stiffness_matrix; i++)
    printf("stiffness_matrix[%d] = %f\n", i, stiffness_matrix[i]);
}
*/

class ElementalRadiationTangentialMatrixAndRadiationLoad: public IntegrationOverEdge{
public:
    void InitializeElementalRadiationTangentialMatrixAndRadiationLoad();
    void set_element_radiation_tangential_matrix_and_radiation_load(int, int, std::vector<int>&, std::vector<double>&,
                                                                    TemperatureDependentVariables*, std::vector<double>&, double);
    void MapElementalToGlobalRadiationTangentialMatrixAndRadiationLoad(PETSC_STRUCT*, std::vector<int>&, int);
    void PrintRadiationTangentialMatrixAndRadiationLoad(std::vector<double>&, std::vector<double>&);
    
private:
    std::vector<std::vector<double> > element_radiation_tangential_matrix_;
    double local_radiation_load[2];
};
void ElementalRadiationTangentialMatrixAndRadiationLoad::InitializeElementalRadiationTangentialMatrixAndRadiationLoad(){
    element_radiation_tangential_matrix_.resize(Constants::kNumOfNodesInElement_);
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        element_radiation_tangential_matrix_[i].resize(Constants::kNumOfNodesInElement_,0.0);
    }
    InitializeIntegrationOverEdge();
}

void ElementalRadiationTangentialMatrixAndRadiationLoad::set_element_radiation_tangential_matrix_and_radiation_load(const int element_number, const int radiation_element_number, std::vector<int>&nodes_in_elements, std::vector<double>& current_temperature_field, TemperatureDependentVariables *const temperature_dependent_variables,std::vector<double>&x_coordinates, const double ambient_temperature){
    //integration rule
    int num_of_integration_points=4;
    double coordinates_of_integration_points[4]={-0.861136312, -0.339981044, 0.339981044, 0.861136312}; //gaussian quadrature coordinates
    double weights_of_integration_points[4]={0.347854845, 0.652145155, 0.652145155, 0.347854845};//weight of gaussian point
    
    //---------zero out element Tangentialradiation matrix--------------
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        for(int j=0;j<Constants::kNumOfNodesInElement_;j++){
            element_radiation_tangential_matrix_[j][i]=0.0;
        }
    }
    
    for(int i=0;i<2;i++)
    local_radiation_load[i]=0.0;
    
    double temperature_quartic_o=pow(ambient_temperature,4);
    
    for(int k=0;k<num_of_integration_points;k++){
        double ksi_coordinate=coordinates_of_integration_points[k];  //gaussian piont coordinate
        double ksi_weight=weights_of_integration_points[k];    //weight of gaussian quadrature
        
        EdgeIntegration(element_number,ksi_coordinate, x_coordinates, nodes_in_elements);
        
        double temperature=0.0;
        temperature += current_temperature_field[(nodes_in_elements[2+element_number*4])]*shape_function_[0];
        temperature += current_temperature_field[(nodes_in_elements[3+element_number*4])]*shape_function_[1];
        
        double temperature_cube=pow(temperature,3);
        double temperature_quartic=pow(temperature,4);
        
        double constant_a = Constants::kStefanBoltzmann_*(*temperature_dependent_variables).get_emissivity(temperature);
//        double constant_a_derivative=Constants::kStefanBoltzmann_*(*temperature_dependent_variables).get_emissivity_derivative(temperature);
        
//        double coefficient=4*constant_a*temperature_cube+constant_a_derivative*temperature_quartic;
        double coefficient = 4*constant_a*temperature_cube;
      
        //note that body heat flux and radiation heat flux are of OPPSITE sign! one increases temperature, while the other one drecreases it.
        element_radiation_tangential_matrix_[2][2]+=coefficient*shape_function_[0]*shape_function_[0]*determinant_of_jacobian_matrix_*ksi_weight;
        element_radiation_tangential_matrix_[2][3]+=coefficient*shape_function_[0]*shape_function_[1]*determinant_of_jacobian_matrix_*ksi_weight;
        element_radiation_tangential_matrix_[3][2]+=coefficient*shape_function_[1]*shape_function_[0]*determinant_of_jacobian_matrix_*ksi_weight;
        element_radiation_tangential_matrix_[3][3]+=coefficient*shape_function_[1]*shape_function_[1]*determinant_of_jacobian_matrix_*ksi_weight;
        
        //calculate radiation load
        local_radiation_load[0] += shape_function_[0]*constant_a*(temperature_quartic-temperature_quartic_o)
        *determinant_of_jacobian_matrix_*ksi_weight;   //3rd node in this element
        local_radiation_load[1] += shape_function_[1]*constant_a*(temperature_quartic-temperature_quartic_o)
        *determinant_of_jacobian_matrix_*ksi_weight;   //4th node in this element
    }
}


void ElementalRadiationTangentialMatrixAndRadiationLoad::MapElementalToGlobalRadiationTangentialMatrixAndRadiationLoad(PETSC_STRUCT* obj, std::vector<int>&equation_numbers_in_elements, const int element_number){
    PetscErrorCode ierr;
    for(int i=0;i<Constants::kNumOfNodesInElement_;i++){
        for(int j=0;j<Constants::kNumOfNodesInElement_;j++){
            int row_equation_number=equation_numbers_in_elements[i+element_number*Constants::kNumOfNodesInElement_];
            int column_equation_number=equation_numbers_in_elements[j+element_number*Constants::kNumOfNodesInElement_];
            if(row_equation_number>=0 && column_equation_number>=0){
//            if(row_equation_number>=0 && column_equation_number>=0 && column_equation_number<=row_equation_number){
                
                PetscErrorCode ierr = MatSetValue(obj->Amat,(PetscInt)row_equation_number,(PetscInt)column_equation_number,(PetscScalar)(element_radiation_tangential_matrix_[i][j]), ADD_VALUES);
                /*
                int position_in_desparsed_matrix=accumulative_half_band_width_vector[row_equation_number]-(row_equation_number-column_equation_number);
                radiation_tangential_matrix[position_in_desparsed_matrix] += element_radiation_tangential_matrix_[i][j];
                 */
            }
        }
    }
    //map to radiation load
    ierr = VecSetValue(obj->rhs,(PetscInt)(equation_numbers_in_elements[element_number*Constants::kNumOfNodesInElement_+2]), (PetscScalar)(-local_radiation_load[0]), ADD_VALUES);
    ierr = VecSetValue(obj->rhs,(PetscInt)(equation_numbers_in_elements[element_number*Constants::kNumOfNodesInElement_+3]), (PetscScalar)(-local_radiation_load[1]), ADD_VALUES);
    /*
    radiation_load[equation_numbers_in_elements[element_number*Constants::kNumOfNodesInElement_+2]] += local_radiation_load[0];
    radiation_load[equation_numbers_in_elements[element_number*Constants::kNumOfNodesInElement_+3]] += local_radiation_load[1];
    */
    //printf("map to global Radiation Tangential Matrix And Radiatio nLoad completed\n");
}

void ElementalRadiationTangentialMatrixAndRadiationLoad::PrintRadiationTangentialMatrixAndRadiationLoad
(std::vector<double>& radiation_tangential_matrix, std::vector<double>& radiation_load){
    int size_of_desparsed_stiffness_matrix = radiation_tangential_matrix.size();
    for(int i=0; i<size_of_desparsed_stiffness_matrix; i++)
    printf("radiation_tangential_matrix[%d] = %f\n", i, radiation_tangential_matrix[i]);
    int size_of_load = radiation_load.size();
    for(int i=0; i<size_of_load; i++)
    printf("radiation_load[%d] = %f\n", i, radiation_load[i]);
}

class Iterations{
public:
    void ZeroVectorAndMatrix(PETSC_STRUCT*);
};
void Iterations::ZeroVectorAndMatrix(PETSC_STRUCT* obj){
    PetscErrorCode ierr;
    ierr = MatZeroEntries(obj->Amat);
    #ifdef radiation
    ierr = MatZeroEntries(obj->stiffness_matrix);
    #endif
    ierr = VecZeroEntries(obj->rhs);
//    ierr = VecZeroEntries(obj->current_temperature_field_local);
}

class OutputResults{
public:
    void OutputVtkFile(Initialization*, GenerateMesh*, std::vector<double>&);
    void OutputCopperSurfaceTemperature(Initialization*, GenerateMesh*, std::vector<double>&);
};
void OutputResults::OutputVtkFile(Initialization *const initialization, GenerateMesh *const generate_mesh, std::vector<double>& current_temperature_field){
    int num_of_nodes=(*((*initialization).get_mesh_parameters())).get_num_of_nodes();
    int dimensions_of_x=(*((*initialization).get_mesh_parameters())).get_dimensions_of_x();
    int dimensions_of_y=(*((*initialization).get_mesh_parameters())).get_dimensions_of_y();
    
    char buffer[30];
    sprintf(buffer, "ModelTemperature.vtk");
    char *filename = buffer;
    FILE *output_vtk_file;
    output_vtk_file = fopen(filename,"w"); //analysis results stored in Output.txt
    if(output_vtk_file==NULL){
        printf("cann't open the file !\n");
        exit(1);
    }
    //  printf("open Output file succeeded\n");
    fprintf(output_vtk_file,"# vtk DataFile Version 2.0\n");
    fprintf(output_vtk_file,"Model Temperature\n");
    fprintf(output_vtk_file,"ASCII\n");
    fprintf(output_vtk_file,"DATASET STRUCTURED_GRID\n"); 
    fprintf(output_vtk_file,"DIMENSIONS %d %d 1\n", dimensions_of_x,dimensions_of_y); 
    fprintf(output_vtk_file,"POINTS %d double\n", num_of_nodes); 
    for(int j=0;j<num_of_nodes;j++){
        fprintf(output_vtk_file,"%.8f  %.8f  0.0\n",(*generate_mesh).get_x_coordinates()[j], (*generate_mesh).get_y_coordinates()[j]);
    }
    fprintf(output_vtk_file, "POINT_DATA %d\n", num_of_nodes);
    fprintf(output_vtk_file, "SCALARS temperature double\n");
    fprintf(output_vtk_file, "LOOKUP_TABLE default\n");
    for(int j=0;j<num_of_nodes;j++){
        fprintf(output_vtk_file,"%.8f\n", current_temperature_field[j]);
    }
    fclose(output_vtk_file);
    printf("writing to vtk file completed......\n");
}

void OutputResults::OutputCopperSurfaceTemperature(Initialization *const initialization, GenerateMesh *const generate_mesh, std::vector<double>& current_temperature_field){
    std::vector<std::pair<double,double> > nodes_on_copper_surface;
    int num_of_nodes=(*((*initialization).get_mesh_parameters())).get_num_of_nodes();
    double y_coordinate_of_copper_surface = (*((*initialization).get_model_geometry())).get_thickness_of_csilicon()
    +(*((*initialization).get_model_geometry())).get_thickness_of_isolater()
    +(*((*initialization).get_model_geometry())).get_thickness_of_titanium()
    +(*((*initialization).get_model_geometry())).get_thickness_of_silicondioxide() 
    +(*((*initialization).get_model_geometry())).get_thickness_of_copper();
    double x_left_bound=(*((*initialization).get_model_geometry())).get_width_of_end(); 
    double x_right_bound=(*((*initialization).get_model_geometry())).get_length_of_model()-(*((*initialization).get_model_geometry())).get_width_of_end();
    double tolerance=1.0e-5;
    
    char buffer[30];
    sprintf(buffer, "SurfaceTemperature.txt");
    char *filename = buffer; 
    FILE *output_copper_surface_temperature;
    output_copper_surface_temperature=fopen(filename,"w"); 
    if(output_copper_surface_temperature==NULL){
        printf("cann't open the file !\n");
        exit(1);
    }
    
    for(int j=0;j<num_of_nodes;j++){
        if( fabs((*generate_mesh).get_y_coordinates()[j]-y_coordinate_of_copper_surface)<tolerance 
           && (*generate_mesh).get_x_coordinates()[j]>x_left_bound-tolerance 
           && (*generate_mesh).get_x_coordinates()[j]<x_right_bound+tolerance ){
            double x_coordinate=(*generate_mesh).get_x_coordinates()[j]
            -(*((*initialization).get_model_geometry())).get_width_of_end();
            double temperature = current_temperature_field[j];
            nodes_on_copper_surface.push_back(std::make_pair(x_coordinate,temperature));
        }
    } 
    
    fprintf(output_copper_surface_temperature,"Surface Temperature\n");
    for(int i=0;i<(nodes_on_copper_surface.size()-1);i++){
        for(int j=i+1;j<nodes_on_copper_surface.size();j++){
            if(nodes_on_copper_surface[i].first>nodes_on_copper_surface[j].first){
                double temporary_variable=nodes_on_copper_surface[i].first;
                nodes_on_copper_surface[i].first=nodes_on_copper_surface[j].first;
                nodes_on_copper_surface[j].first=temporary_variable;
                temporary_variable=nodes_on_copper_surface[i].second; 
                nodes_on_copper_surface[i].second=nodes_on_copper_surface[j].second;
                nodes_on_copper_surface[j].second=temporary_variable;
            }
        }
        fprintf(output_copper_surface_temperature,"%.8f\t%.10f\t\n",nodes_on_copper_surface[i].first,nodes_on_copper_surface[i].second);
    }
    fprintf(output_copper_surface_temperature,"%.8f\t%.10f\t\n",nodes_on_copper_surface[(nodes_on_copper_surface.size()-1)].first,
            nodes_on_copper_surface[(nodes_on_copper_surface.size()-1)].second);
    
    fclose(output_copper_surface_temperature);
    std::cout<<"writing copper surface temperature completed\n"<<std::endl;
}

int main(int argc, char **args) {
    int rank, size;
    MPI_Init (&argc, &args);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
//    MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */

    static char help[]="Solves a tri-diagonal system using KSP.\n";
    PetscErrorCode ierr;
    PetscViewer viewer;
    PetscScalar *get;
    PETSC_STRUCT sys;
    VecScatter ctx;
    PETSC_VEC current_temperature_field_global;
    PetscScalar error_norm = 0.0;
    PetscScalar rhs_norm = 0.0;
    PetscInt equation_number_local_start, equation_number_local_end_plus_one, row_local_start, row_local_end_plus_one;

    /* Initialize PETSc */
    Petsc_Init(argc, args, help);

    if(rank == 0) {
        printf("\n\n******************************************************************************************\n");
        printf("*************************** Debugged on PETSC2.3.3 *********************************************\n");
        printf("********************************************************************************************");
        printf("\n\n\t*****Heat Transfer Simulation for Real Time Grain Growth Control of Copper Film*****\n");
        printf("\tThis code is developed for the project 'Real Time Control of Grain Growth in Metals' (NSF reference codes: 024E, 036E, 8022, AMPP)\n\n");
        printf("\tProject Investigators:\n");
        printf("\tRobert Hull hullr2@rpi.edu (Principal Investigator)\n\tJohn Wen (Co-Principal Investigator)\n\tAntoinette Maniatty (Co-Principal Investigator)\n\tDaniel Lewis (Co-Principal Investigator)\n\n");
        printf("\tCode developer: Yixuan Tan tany3@rpi.edu\n\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    Initialization initialization;
    initialization.InitializeInitialization();
    double ambient_temperature=(*(initialization.get_analysis_constants())).get_ambient_temperature();
    double initial_time_increment=(*(initialization.get_analysis_constants())).get_initial_time_increment();
    double minimum_time_increment=(*(initialization.get_analysis_constants())).get_minimum_time_increment();
    int maximum_time_steps=(*(initialization.get_analysis_constants())).get_maximum_time_steps();
    double total_simulation_time=(*(initialization.get_analysis_constants())).get_total_simulation_time();
    double maximum_temperature_change_per_time_increment=(*(initialization.get_analysis_constants())).get_maximum_temperature_change_per_time_increment();
    double time_to_turn_off_heaters=(*(initialization.get_analysis_constants())).get_time_to_turn_off_heaters();
    int output_time_step_interval=(*(initialization.get_analysis_constants())).get_output_time_step_interval();
    int num_of_nodes = (*(initialization.get_mesh_parameters())).get_num_of_nodes();
    int num_of_elements = (*(initialization.get_mesh_parameters())).get_num_of_elements();
    double time_increment=initial_time_increment;
    int num_of_iterations_with_unchanged_time_increment=0;
    int iteration_number=0;
    double current_time=0.0;
    bool check_temperature_change_size_satisfiable;
    bool is_heaters_turned_off=false;
    double temperature_norm_last = 0.0;
    double temperature_norm_current = 0.0;
    
    //read input data to generate mesh
    GenerateMesh generate_mesh;
    generate_mesh.GenerateMeshInitializeMeshSizeInfo(&initialization);
    generate_mesh.CalculateCoordinates(&initialization);
    //  generate_mesh.PrintCoordinatesResults(&initialization);
    std::vector<double> &x_coordinates = generate_mesh.get_x_coordinates();
    std::vector<double> &y_coordinates = generate_mesh.get_y_coordinates();
    
    DegreeOfFreedomAndEquationNumbers dof_and_equation_numbers;
    dof_and_equation_numbers.InitializeDegreeOfFreedomAndEquationNumbers(&initialization);
    dof_and_equation_numbers.set_essential_bc_nodes(&initialization, y_coordinates);
    dof_and_equation_numbers.GenerateNodeAndEquationNumbersInElements(&initialization);
    //  dof_and_equation_numbers.PrintDofAndEquationNumbers(&initialization);

    std::vector<int> &nodes_in_elements = dof_and_equation_numbers.get_nodes_in_elements();
    int num_of_essential_bc_nodes = dof_and_equation_numbers.get_num_of_essential_bc_nodes();
    std::vector<int> &essential_bc_nodes = dof_and_equation_numbers.get_essential_bc_nodes();
    int num_of_equations = dof_and_equation_numbers.get_num_of_equations();
    std::vector<int> &equation_numbers_of_nodes = dof_and_equation_numbers.get_equation_numbers_of_nodes();
    std::vector<int> &equation_numbers_in_elements = dof_and_equation_numbers.get_equation_numbers_in_elements();
  
    TemperatureDependentVariables temperature_dependent_variables;
    temperature_dependent_variables.InitializeTemperatureDependentVariables(&initialization);
    
    BoundaryCondition boundary_condition;
    boundary_condition.InitializeBoundaryCondition(&initialization);
    
    //  boundary_condition.PrintBoundaryConditionNodes(essential_bc_nodes);
    
    HeaterElements heater_elements;
    heater_elements.InitializeHeaterElements(&initialization);
    heater_elements.set_elements_as_heater(&initialization);
    //  heater_elements.PrintHeaterElements();
    int num_of_elements_as_heater=heater_elements.get_num_of_elements_as_heater();
    std::vector<int>&elements_as_heater=heater_elements.get_elements_as_heater();
    
    RadiationElements radiation_elements;
    radiation_elements.InitializeRadiationElements(&initialization);
    radiation_elements.set_elements_with_radiation(&initialization);
    //  radiation_elements.PrintRadiationElements();
    int num_of_elements_with_radiation=radiation_elements.get_num_of_elements_with_radiation();
    std::vector<int> &elements_with_radiation=radiation_elements.get_elements_with_radiation();
    
    MaterialParameters material_parameters;
    material_parameters.set_densities();
    material_parameters.set_material_id_of_elements(&initialization);
    //  material_parameters.PrintMaterialParameters();
    std::vector<int>& material_id_of_elements = material_parameters.get_material_id_of_elements();
    std::vector<double>& densities = material_parameters.get_densities();
   
    MPI_Barrier(MPI_COMM_WORLD);
    /*Create the vectors and matrix we need for PETSc*/
    Vec_Create(&sys,(PetscInt)num_of_equations);
    //            std::cout<<"vec_creation finished \n";
    Mat_Create(&sys,(PetscInt)num_of_equations,(PetscInt)num_of_equations);
    //            std::cout<<"mat_creation finished \n";
    MPI_Barrier(MPI_COMM_WORLD);

    /*
    GlobalVectorsAndMatrices global_vectors_and_matrices;
    global_vectors_and_matrices.InitializeGlobalVectorsAndMatrices(num_of_nodes, accumulative_half_band_width_vector);
    std::vector<double>& stiffness_matrix = global_vectors_and_matrices.get_stiffness_matrix();
    std::vector<double>& mass_matrix = global_vectors_and_matrices.get_mass_matrix();
//    std::vector<double>& radiation_tangential_matrix = global_vectors_and_matrices.get_radiation_tangential_matrix(); // do not need tangentail any more
//    std::vector<double>& body_heat_flux_tangential_matrix = global_vectors_and_matrices.get_body_heat_flux_tangential(); // do not need tangentail any more
//    std::vector<double>& heat_load = global_vectors_and_matrices.get_heat_load();  // do not need, this goes directly into rhs
//    std::vector<double>& radiation_load = global_vectors_and_matrices.get_radiation_load(); // do not need, this goes directly into rhs
    std::vector<double>& current_temperature_field = global_vectors_and_matrices.get_current_temperature_field();
    std::vector<double>& right_hand_side_function = global_vectors_and_matrices.get_right_hand_side_function();
//    std::vector<double>& jacobian_matrix_global = global_vectors_and_matrices.get_jacobian_matrix_global();
    std::vector<double>& solution_increments_trial = global_vectors_and_matrices.get_solution_increments_trial();
    std::vector<double>& initial_temperature_field = global_vectors_and_matrices.get_initial_temperature_field();
    std::vector<double>& solution_of_last_iteration = global_vectors_and_matrices.get_solution_of_last_iteration();
    */
    
    std::vector<double> current_temperature_field;
    current_temperature_field.resize(num_of_nodes, 0.0);
    
    ElementalMatrix elemental_matrix;
    elemental_matrix.InitializeElementalMatrix();
    
//    ElementalBodyHeatFluxTangentialMatrix elemental_body_heat_flux_tangential_matrix;
//    elemental_body_heat_flux_tangential_matrix.InitializeElementalBodyHeatFluxTangentialMatrix();
    ElementalRadiationTangentialMatrixAndRadiationLoad elemental_radiation_tangential_matrix_and_radiation_load;
    elemental_radiation_tangential_matrix_and_radiation_load.InitializeElementalRadiationTangentialMatrixAndRadiationLoad();
    
//    Assemble assemble;
    Iterations iterations;
    OutputResults output_results;
//    std::cout<<"1111\n";
//    getchar();
   
/*
    // NOTE current_temperature_field_local does not go back to zero after every iteration
    ierr = VecCreate(PETSC_COMM_WORLD, &(sys.current_temperature_field_local));
    ierr = VecSetSizes(sys.current_temperature_field_local, PETSC_DECIDE, num_of_equations);
    ierr = VecSetFromOptions(sys.current_temperature_field_local);
*/
    /*Create the vectors and matrix we need for PETSc*/
    Vec_Create(&sys,(PetscInt)num_of_equations);
    //           std::cout<<"vec_creation finished \n";
    Mat_Create(&sys, (PetscInt)num_of_equations, (PetscInt)num_of_equations);
    //            std::cout<<"mat_creation finished \n";
    MPI_Barrier(MPI_COMM_WORLD);
    
    TemperatureFieldInitial temperature_field_initial;
    // set the initial temperature field, which is to initialize the current_temperature_field_local and the double vector: current_temperature_field_local
    temperature_field_initial.set_initial_temperature_field(essential_bc_nodes, current_temperature_field, &sys, &initialization, equation_numbers_of_nodes);
    MPI_Barrier(MPI_COMM_WORLD);

    // get local ownership range
    ierr = MatGetOwnershipRange(sys.Amat, &equation_number_local_start, &equation_number_local_end_plus_one);
    ierr = VecGetOwnershipRange(sys.sol, &row_local_start, &row_local_end_plus_one);
    
    PetscInt *d_nnz = new PetscInt[equation_number_local_end_plus_one - equation_number_local_start];
    PetscInt *o_nnz = new PetscInt[equation_number_local_end_plus_one - equation_number_local_start];

    EachRowNozeroCount each_row_nozero_count;

    MPI_Barrier(MPI_COMM_WORLD);
    each_row_nozero_count.SetDnnzAndOnnz(&initialization, &dof_and_equation_numbers, (int)equation_number_local_start, (int)equation_number_local_end_plus_one, d_nnz, o_nnz);
    
    MPI_Barrier(MPI_COMM_WORLD);
    // once we have ownership range, we can preallocate memory for the matrices by providing d_nnz and o_nnz
    Mat_Preallocation(&sys, d_nnz, o_nnz);
    /*
    if(rank == 1)
        std::cout << equation_number_local_start <<" "<<equation_number_local_end_plus_one << "\n";
    if(rank == 1)
        std::cout << row_local_start <<" "<< row_local_end_plus_one << "\n";
    */

    while(1){ // this while loop governs the iterations to solution
        
        ++iteration_number;
     
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int element_number=0; element_number < num_of_elements; element_number++) {
            // check if there is at least one equation number in this element that lives in this processor
            for (int node_count_inside = 1; node_count_inside < Constants::kNumOfNodesInElement_; node_count_inside++){
                int equation_count = equation_numbers_in_elements[node_count_inside + element_number*Constants::kNumOfNodesInElement_];
                if(equation_count >= equation_number_local_start && equation_count < equation_number_local_end_plus_one){
                
                    // okay, you are in my land, I have to take care of you
                    elemental_matrix.set_coordinates_in_this_element(element_number, nodes_in_elements, x_coordinates, y_coordinates);
                    elemental_matrix.set_element_stiffness_matrix(element_number, nodes_in_elements, material_id_of_elements, current_temperature_field, &temperature_dependent_variables);
                    elemental_matrix.MapElementalToGlobalStiffness(&sys, equation_numbers_in_elements, element_number);
                    std::vector<std::vector<double> >&element_stiffness_matrix = elemental_matrix.get_element_stiffness_matrix();
                    boundary_condition.FixTemperature(element_number, element_stiffness_matrix, equation_numbers_in_elements, &sys);
                    
                    // okay, let's go to next element
                    break; // break from "node_count_inside"
                }
            }

        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int heater_element_number=0; heater_element_number < num_of_elements_as_heater; heater_element_number++){
            int element_number = elements_as_heater[heater_element_number];
            
            // check if there is at least one equation number in this element that lives in this processor
            for (int node_count_inside = 1; node_count_inside < Constants::kNumOfNodesInElement_; node_count_inside++){
                int equation_count = equation_numbers_in_elements[node_count_inside + element_number*Constants::kNumOfNodesInElement_];
                if(equation_count >= row_local_start && equation_count < row_local_end_plus_one){
                    
                    // okay, you are in my land, I have to take care of you
                    heater_elements.set_coordinates_in_this_element(element_number, nodes_in_elements, x_coordinates, y_coordinates);
                    heater_elements.HeatSupply(element_number, heater_element_number, &sys, nodes_in_elements, equation_numbers_in_elements, current_temperature_field, &temperature_dependent_variables, &initialization);

                    // okay, let's go to next element
                    break; // break from "node_count_inside"
                }
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        #ifdef radiation
        //    elemental_body_heat_flux_tangential_matrix.PrintBodyHeatFluxTangentialMatrix(body_heat_flux_tangential_matrix);
        for(int radiation_element_number=0; radiation_element_number < num_of_elements_with_radiation; radiation_element_number++) {
            int element_number = elements_with_radiation[radiation_element_number];
            
            // check if there is at least one equation number in this element that lives in this processor
            for (int node_count_inside = 1; node_count_inside < Constants::kNumOfNodesInElement_; node_count_inside++){
                int equation_count = equation_numbers_in_elements[node_count_inside + element_number*Constants::kNumOfNodesInElement_];
                if(equation_count >= row_local_start && equation_count < row_local_end_plus_one){
                    
                    // okay, you are in my land, I have to take care of you
                    elemental_radiation_tangential_matrix_and_radiation_load.set_element_radiation_tangential_matrix_and_radiation_load(element_number, radiation_element_number, nodes_in_elements, current_temperature_field, &temperature_dependent_variables, x_coordinates, ambient_temperature);
                    elemental_radiation_tangential_matrix_and_radiation_load.MapElementalToGlobalRadiationTangentialMatrixAndRadiationLoad(&sys, equation_numbers_in_elements, element_number);

                    // okay, let's go to next element
                    break; // break from "node_count_inside"
                }
            }
        }
        
        //      elemental_radiation_tangential_matrix_and_radiation_load.PrintRadiationTangentialMatrixAndRadiationLoad(radiation_tangential_matrix, radiation_load);
        
//        std::cout<<"2222\n";
//        getchar();
        
        // assemble the stiffness matrix first, because stiffness matrix will be used to modifiy the rhs
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
        
//            ierr = MatAssemblyBegin(sys.stiffness_matrix, MAT_FINAL_ASSEMBLY);
//            ierr = MatAssemblyEnd(sys.stiffness_matrix, MAT_FINAL_ASSEMBLY);
            //Indicate same nonzero structure of successive linear system matrices
//            MatSetOption(sys.stiffness_matrix, MAT_NO_NEW_NONZERO_LOCATIONS);
            
        Petsc_Assem_Matrices(&sys);
//            std::cout<<"3333\n";
//            getchar();
            
        MPI_Barrier(MPI_COMM_WORLD);
            
        Petsc_Assem_Vectors(&sys);
//            std::cout<<"44444\n";
//            getchar();

        MPI_Barrier(MPI_COMM_WORLD);
//        std::cout<<"55555\n";
//        getchar();
      
        
        #ifdef radiation
        // subtract stiffness_matrix * current_temperature_field from the rhs.
        // note here the stiffness matrix has already been added a negative sign on every component. This because we need to subtract the stiffness_matrix(original) * current_temperature_field
        ierr = MatMultAdd(sys.stiffness_matrix, sys.current_temperature_field_local, sys.rhs, sys.rhs);
        
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
        
//        std::cout<<"66666\n";
//        getchar();
    
        /* Call function to do final assembly of PETSc matrix and vectors*/
 //       Petsc_Assem(&sys);
        
//        MPI_Barrier(MPI_COMM_WORLD);
        
//        if (iteration_number == 1){
//            //Indicate same nonzero structure of successive linear system matrices
//            MatSetOption(sys.Amat, MAT_NO_NEW_NONZERO_LOCATIONS);
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
        
        /* Call function to solve the tri-diagonal syste*/
        Petsc_Solve(&sys);
        
        // calculate the norm of the solution and norm of the rhs.
        MPI_Barrier(MPI_COMM_WORLD);
//        std::cout<<"77777\n";
//        getchar();
        #ifdef radiation
        ierr = VecNorm(sys.sol, NORM_2, &error_norm);
        ierr = VecNorm(sys.rhs, NORM_2, &rhs_norm);
        MPI_Barrier(MPI_COMM_WORLD);
        // add the solution to the current_temperature_field_local
        ierr = VecAXPY(sys.current_temperature_field_local, 1, sys.sol);
        #else // no radiation boundary condition
        // subtract the current_temperature_field_local by the solution to get the error
        ierr = VecAXPY(sys.current_temperature_field_local, -1, sys.sol);
        ierr = VecNorm(sys.current_temperature_field_local, NORM_2, &error_norm);
        MPI_Barrier(MPI_COMM_WORLD);
        // restore the current_temperature_field_local
        ierr = VecCopy(sys.sol, sys.current_temperature_field_local);
        #endif

        // std::cout<<"error_norm is "<<error_norm<<"\n";
        
        // scatter the sys.current_temperature_field_local to the global current_temperature_field
        ierr = VecScatterCreateToAll(sys.current_temperature_field_local, &ctx, &current_temperature_field_global);
        ierr = VecScatterBegin(ctx, sys.current_temperature_field_local, current_temperature_field_global, INSERT_VALUES, SCATTER_FORWARD);
        ierr = VecScatterEnd(ctx, sys.current_temperature_field_local, current_temperature_field_global, INSERT_VALUES, SCATTER_FORWARD);
        
        // Call function to get a pointer to the global current_temperature_field vector*/
        ierr = VecGetArray(current_temperature_field_global, &get);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Update the current_temperature_field vector on each node.
        for(int j=0; j<num_of_nodes; j++){
            int equation_count = equation_numbers_of_nodes[j];
            if(equation_count>=0)
            current_temperature_field[j] = get[equation_count];
        }
        
        /* Free PETSc objects */
        ierr = VecRestoreArray(current_temperature_field_global, &get);
        ierr = VecScatterDestroy(ctx);
        ierr = VecDestroy(current_temperature_field_global);
        
        // check for convergence
        /*
        if(rank == 0){
            std::cout<<"error_norm: "<<(double)error_norm<<"  rhs_norm: "<<(double)rhs_norm<<"\n";
        }
        */
        #ifdef radiation
        if((double)error_norm < Constants::kNormTolerance_ && (double)rhs_norm < Constants::kYFunctionTolerance_){// convergence must be satisfied first, then consider temperature increment size.
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                printf("number of iteration to converge is %d\n", iteration_number);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            break;  // break from the while loop
        }
        #else
        if((double)error_norm < Constants::kNormTolerance_ ){// convergence must be satisfied first, then consider temperature increment size.
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                printf("number of iteration to converge is %d\n", iteration_number);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            break;  // break from the while loop
        }
        #endif
        
        //zero all necessary vectors and matrices for next iteration
        iterations.ZeroVectorAndMatrix(&sys);

    }//while
    Petsc_Destroy(&sys);
    //Must destroy current_temperature_field_local
//    ierr = VecDestroy(sys.current_temperature_field_local);
    
    /* Free all PETSc objects created for solve */
//    Petsc_Destroy(&sys);
   
    if(rank==0){
        output_results.OutputVtkFile(&initialization, &generate_mesh, current_temperature_field);
        output_results.OutputCopperSurfaceTemperature(&initialization, &generate_mesh, current_temperature_field);

        FILE* current_densities;
        current_densities=fopen("current_densities.txt","w");
        //  fprintf(current_densities,"units are A/m^2\n");
        fprintf(current_densities, "units are A/cm^2\n");
        //  double heater_cross_section_area=temperature_dependent_variables.get_heater_crosssection_area_mm_square()*1.0e-3*1.0e-3; //mm^2 -> m^2
        double heater_cross_section_area = temperature_dependent_variables.get_heater_crosssection_area_mm_square()*1.0e-1*1.0e-1; //mm^2 -> cm^2
        for(int i=0;i<Constants::kNumOfHeaters_;i++){
            double current=(*(initialization.get_currents_in_heater())).get_current_in_heater()[i]*1.0e-3; //mA -> A
            fprintf(current_densities,"%e\n",current/heater_cross_section_area);
        }
        fclose(current_densities);
        printf("Analysis completed successfully!\n");
        printf(" a (model temperature field).vtk file, a (copper surface temperature).txt file and a (current_density).txt file have been generated\n\n");
        
        std::cout<<"\n number of equation is "<< num_of_equations << std::endl;
    }
    
    
    delete [] d_nnz; d_nnz = NULL;
    delete [] o_nnz; o_nnz = NULL;
  
    MPI_Barrier(MPI_COMM_WORLD);
    /*finalize PETSc*/
    Petsc_End();
    MPI_Finalize();
    return 0;
}
