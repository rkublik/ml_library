# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:09:57 2017

@author: Richard
"""

import matplotlib.pyplot as plt
from math import cos, sin, atan

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def draw(self, neuron_radius):
        circle = plt.Circle((self.x, self.y), radius = neuron_radius, fill = False)
        plt.gca().add_patch(circle)
    
    
class Layer():
    def __init__(self, network,number_of_neurons, number_of_neurons_in_widest_layer):
        self.horizontal_distance_between_layers = 6
        self.vertical_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__initialize_neurons(number_of_neurons)
        
    def __initialize_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_bottom_margin_so_layer_is_centered(number_of_neurons)
        for iteration in xrange(number_of_neurons):
            neuron = Neuron(self.x, y)
            neurons.append(neuron)
            y+= self.vertical_distance_between_neurons
        return neurons
    
    def __calculate_bottom_margin_so_layer_is_centered(self, number_of_neurons):
        return self.vertical_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2
    
    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + self.horizontal_distance_between_layers
        else:
            return 0
        
    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None
        
    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.y - neuron1.y)/float(neuron2.x - neuron1.x))
        
        x_adjustment = self.neuron_radius * cos(angle)
        y_adjustment = self.neuron_radius * sin(angle)

        line = plt.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                          (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        plt.gca().add_line(line)
        
    def draw(self, layerType = 0):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
                    
        # Write text:
        if layerType == 0:
            plt.text(self.x, -1, 'Input Layer', fontsize = 12, rotation = -45)
        elif layerType == -1:
            plt.text(self.x, -1, 'Output Layer', fontsize = 12, rotation = -45)
        else:
            plt.text(self.x, -1, 'Hidden Layer %d' % layerType, fontsize = 12, rotation = -45)
           
            
class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertypes = 0
        
    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)
        
    def draw(self, plot_title = "Neural Network Architecture", fig = None):
        if fig is None:
            fig = plt.figure()
        else: 
            plt.set_current_figure(fig)
            
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw(i)
        plt.axis("tight")
        plt.axis('off')
        plt.title(plot_title, fontsize = 15)
        plt.show()
        return fig
        
class DrawNN():
    def __init__(self, neural_network):
        self.neural_network = neural_network
        
    def draw(self, title = "Neural Network Architecture", fig = None):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        for l in self.neural_network:
            network.add_layer(l)
        f = network.draw(title, fig)
        
        
if __name__ == "__main__":
    network = DrawNN([20,14,8,10,10,9,15])
    network.draw()