using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Number_Recognizer_CNN.Core;

namespace Number_Recognizer_CNN.Neural_Network
{

    public class Layer
    {
        private LayerType layerType;
        public LayerType LayerType
        {
            get { return layerType; }
        }
        private Neuron[] _neurons;
        public Neuron[] Neurons
        {
            get { return _neurons; }
            set { _neurons = value; }
        }
        private Layer _connectedLayer;

        public Layer ConnectedLayer
        {
            get { return _connectedLayer; }
            set { _connectedLayer = value; }
        }

        public Layer(int number_of_neurons, LayerType type, ActivationType activationType)
        {
            _neurons = new Neuron[number_of_neurons];
            for (int i = 0; i < _neurons.Length; i++)
            {
                _neurons[i] = new Neuron(this, activationType);
            }
            layerType = type;
        }
        public void InitInputNeurons(double[] inputValues)
        {
            for (int i = 0; i < _neurons.Length; i++)
            {
                _neurons[i].Activation = inputValues[i];
            }
        }
        public void SetConnectedLayers(Layer layer)
        {
            this.ConnectedLayer = layer;
            for (int i = 0; i < this._neurons.Length; i++)
            {
                this._neurons[i].ConnectNeurons(layer);
            }
        }

    }
}
