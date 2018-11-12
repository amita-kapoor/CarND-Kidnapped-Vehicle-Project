/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#define epsilon 0.00001 

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    // Number of particles: 

	if (initialized()) return; // Incase already initialized no need to redo it
	num_particles = 50; 

	default_random_engine gen;

	// Create normal distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize all particles to first position 
	for (int i = 0; i<num_particles; i++) {
	  Particle particle;
	  particle.id = i;
	  particle.x = dist_x(gen);
	  particle.y = dist_y(gen);
	  particle.theta = dist_theta(gen);
	  particle.weight = 1.0;
	  
	  particles.push_back(particle);
	  weights.push_back(particle.weight);
	} // End of initialization of particles

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
	default_random_engine gen;
	// Creating normal distributions
    normal_distribution<double> dist_x(0, std_pos[0]);  //x
    normal_distribution<double> dist_y(0, std_pos[1]);  //y 
    normal_distribution<double> dist_theta(0, std_pos[2]);  //theta

	// Calculate new state for each particle
	for (int i = 0; i<num_particles; i++){ 
		if (fabs(yaw_rate) > epsilon) {
			particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      		particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      		particles[i].theta += yaw_rate * delta_t;

		} // end if true
		else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} // end else

		// Add Noise
		particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
	} // end for
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
    
    	// Get the current landmark observation
        LandmarkObs obs = observations[i];

    	// initialize the minimum distance to maximum possible value in the machine
    	double min_dist = numeric_limits<double>::max();

    	// id of landmark from map placeholder to be associated with the observation
    	int closest_landmark_id = -1;
    
    	for (unsigned int j = 0; j < predicted.size(); j++) {
      		// Get current prediction
      		LandmarkObs pred = predicted[j];
      
      		// The distance between current/predicted landmarks
      		double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

      		// Find the predicted landmark nearest the current observed landmark
      		if (cur_dist < min_dist) {
        		min_dist = cur_dist;
        		closest_landmark_id = pred.id;
      		}
        }

    	// Set the observation's id to the nearest predicted landmark's id
    	observations[i].id = closest_landmark_id;
  	}



}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    // First term of multi-variate normal Gaussian distribution calculated below
    // It stays the same so can be outside the loop
    const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
    // The denominators of the mvGd also stay the same
    const double x_denom = 2 * std_landmark[0] * std_landmark[0];
    const double y_denom = 2 * std_landmark[1] * std_landmark[1];

    // Iterate through each particle
    for (int i = 0; i < num_particles; ++i) {
    
       // For calculating multi-variate Gaussian distribution of each observation, for each particle
       double mvGd = 1.0;
    
       // For each observation
       for (unsigned int j = 0; j < observations.size(); ++j) {
      
        // Transform the observation point (from vehicle coordinates to map coordinates)
        double trans_obs_x, trans_obs_y;
        trans_obs_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
        trans_obs_y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
      
        // Find nearest landmark
        vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
        vector<double> landmark_obs_dist (landmarks.size());
        for (unsigned int k = 0; k < landmarks.size(); ++k) {
        
          // Down-size possible amount of landmarks to look at by only looking at those in sensor range of the particle
          // If in range, put in the distance vector for calculating nearest neighbor
          double landmark_part_dist = sqrt(pow(particles[i].x - landmarks[k].x_f, 2) + pow(particles[i].y - landmarks[k].y_f, 2));
          if (landmark_part_dist <= sensor_range) 
             landmark_obs_dist[k] = sqrt(pow(trans_obs_x - landmarks[k].x_f, 2) + pow(trans_obs_y - landmarks[k].y_f, 2));
         else 
            // Need to fill those outside of distance with huge number, or they'll be a zero (and think they are closest)
            landmark_obs_dist[k] = 999999.0;
        }
      
        // Associate the observation point with its nearest landmark neighbor
        int min_pos = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
        float nn_x = landmarks[min_pos].x_f;
        float nn_y = landmarks[min_pos].y_f;
      
        // Calculate multi-variate Gaussian distribution
        double x_diff = trans_obs_x - nn_x;
        double y_diff = trans_obs_y - nn_y;
        double b = ((x_diff * x_diff) / x_denom) + ((y_diff * y_diff) / y_denom);
        mvGd *= a * exp(-b);
      
        }  //end for each observation
    
      // Update particle weights with combined multi-variate Gaussian distribution
      particles[i].weight = mvGd;
      weights[i] = particles[i].weight;

    }  //end eacxh particle

} //end update weights

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<double> weights;
	default_random_engine gen;

	double maxWeight = numeric_limits<double>::min();  // Max limit for weights

	// Get weights
	for(int i = 0; i < num_particles; i++) {
    	weights.push_back(particles[i].weight);
    	if ( particles[i].weight > maxWeight ) 
      		maxWeight = particles[i].weight;
    }

	// Creating distributions.
    uniform_real_distribution<double> distDouble(0.0, maxWeight);
    uniform_int_distribution<int> distInt(0, num_particles - 1);

    // Generating index.
    int index = distInt(gen);

    double beta = 0.0;

    // the wheel
    vector<Particle> resample_particles;
    for(int i = 0; i < num_particles; i++) {
        beta += distDouble(gen) * 2.0;
        while( beta > weights[index]) {
           beta -= weights[index];
           index = (index + 1) % num_particles;
        }
        resample_particles.push_back(particles[index]);
  }

  // Replace old particles with the resampled particles
  particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
