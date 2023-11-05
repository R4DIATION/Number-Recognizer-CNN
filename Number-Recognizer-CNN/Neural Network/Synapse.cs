using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Number_Recognizer_CNN.Neural_Network
{
    public class Synapse
    {
		private Neuron _inputNeuron;

		public Neuron InputNeuron
		{
			get { return _inputNeuron; }
			set { _inputNeuron = value; }
		}

		private Neuron _outputNeuron;

		public Neuron OutputNeuron
		{
			get { return _outputNeuron; }
			set { _outputNeuron = value; }
		}

		private double _weight;

		public double Weight
		{
			get { return _weight; }
			set { _weight = value; }
		}

        public Synapse(Neuron input, Neuron output)
        {
            this._inputNeuron = input;
			this._outputNeuron = output;
			this.Weight = Core.rnd.NextDouble() * 2 - 1;
        }
    }
}
