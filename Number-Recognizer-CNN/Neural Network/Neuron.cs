using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Number_Recognizer_CNN.Core;

namespace Number_Recognizer_CNN.Neural_Network
{
    public class Neuron
    {
        public Layer ContainingLayer {  get; private set; }
        private double[] _gradient;

        public double[] Gradient
        {
            get { return _gradient; }
            set { _gradient = value; }
        }

        private ActivationType _activationFunc;
        private double _activation;
        
        public double Activation
        {
            get { return _activation; }
            set { _activation = value; }
        }

        private Synapse[] _synapses;
        public Synapse[] Synapses
        {
            get { return _synapses; }
            set { _synapses = value; }
        }

        private double _bias;
        public double Bias
        {
            get { return _bias; }
            set { _bias = value; }
        }

        private double _weightedSum;

        public double WeightedSum
        {
            get { return _weightedSum; }
            set { _weightedSum = value; }
        }

        public Neuron(Layer layer, ActivationType activation)
        {
            this.ContainingLayer = layer;
            _activationFunc = activation;
            this.Activation = 0;
            this.Bias = 1.0;
        }
        public void ConnectNeurons(Layer PrevLayer)
        {
            _synapses = new Synapse[PrevLayer.Neurons.Length];
            for (int i = 0; i < _synapses.Length; i++)
            {
                _synapses[i] = new Synapse(PrevLayer.Neurons[i], this);
            }
        }
        public void CalculateActivation()
        {
            _weightedSum = 0;
            for (int i = 0; i < Synapses.Length; i++)
            {
                WeightedSum = Synapses[i].Weight * Synapses[i].InputNeuron.Activation;
            }
            this.Activation = Sigmoid(WeightedSum + Bias);
        }
        public void CalculateGradient(double? expected = null)
        {
            if (ContainingLayer.LayerType == LayerType.Output && expected != null)
            {
                Gradient = new double[Synapses.Length];
                for (int i = 0; i < Synapses.Length; i++)
                {
                    Gradient[i] = (double)(Synapses[i].InputNeuron.Activation * (2 * (this.Activation - expected)));
                }
            }
            else if (ContainingLayer.LayerType == LayerType.Hidden)
            {
                Gradient = new double[Synapses.Length];
                for (int i = 0; i < Synapses.Length; i++)
                {
                    Gradient[i] = Synapses[i].Weight * Synapses[i].InputNeuron.Activation * (1 - Synapses[i].InputNeuron.Activation);
                }
            }
        }
    }
}
