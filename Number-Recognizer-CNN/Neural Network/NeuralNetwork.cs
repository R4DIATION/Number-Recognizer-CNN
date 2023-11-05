using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Number_Recognizer_CNN.Core;

namespace Number_Recognizer_CNN.Neural_Network
{
   
    public class NeuralNetwork
    {
        private double _cost;
        private int HiddenNeuronCount;
        private int _expectedValue;
        private Layer[] layers;
        public NeuralNetwork(int number_of_hidden_neurons)
        {
            this.HiddenNeuronCount = number_of_hidden_neurons;
            layers = new Layer[0];
        }
        public void InitNeuralNetwork(double[] inputNumbers, int number_of_input_neurons, int expected_value)
        {
            this._expectedValue = expected_value;
            this.layers = new Layer[3];
            this.layers[0] = new Layer(number_of_input_neurons, LayerType.Input, ActivationType.None);
            this.layers[1] = new Layer(this.HiddenNeuronCount, LayerType.Hidden, ActivationType.Sigmoid);
            this.layers[2] = new Layer(10, LayerType.Output, ActivationType.None);

            this.layers[0].InitInputNeurons(inputNumbers);
            ConnectLayers();
            CalculateActivation();
            Softmax();
            CalculateCost();

            if (_cost != 0)
            {
                Backpropagate();
            }
        }
        private void ConnectLayers()
        {
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].SetConnectedLayers(layers[i - 1]);
            }
        }
        private void CalculateActivation()
        {
            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].Neurons.Length; j++)
                {
                    layers[i].Neurons[j].CalculateActivation();
                }
            }
        }
        private void Softmax()
        {
            double[] helper = new double[layers.Last().Neurons.Length];

            for (int i = 0; i < helper.Length; i++)
            {
                helper[i] = layers.Last().Neurons[i].Activation * Math.E;
            }

            for (int i = 0; i < helper.Length; i++)
            {
                helper[i] = i / helper.Sum();
            }

            for (int i = 0; i < helper.Length; i++)
            {
                layers.Last().Neurons[i].Activation = helper[i];
            }
        }
        private void CalculateCost()
        {
            double max = layers.Last().Neurons.Max(x => x.Activation);
            int selectedNumber = 0;
            for (int i = 0; i < layers.Last().Neurons.Length; i++)
            {
                if (max == layers.Last().Neurons[i].Activation)
                {
                    selectedNumber = i;
                }
            }
            _cost = Cost(selectedNumber, this._expectedValue);

            Console.WriteLine($"Expected number: {_expectedValue}");
            Console.WriteLine($"Actual number: {selectedNumber}");
            Console.WriteLine($"C O S T: {_cost}");
        }
        private void Backpropagate()
        {

        }
    }
    
}
