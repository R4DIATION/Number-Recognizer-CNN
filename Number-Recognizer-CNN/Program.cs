using Number_Recognizer_CNN.Convolution;
using Number_Recognizer_CNN.Neural_Network;

namespace Number_Recognizer_CNN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int number_of_samples = 1;
            NeuralNetwork network = new NeuralNetwork(16);
            List<string> list = File.ReadAllLines("numbers.csv").Skip(1).Select(x => x).ToList();
            for (int i = 0; i < number_of_samples; i++)
            {
                string[] helper = list[i].Split(',');
                convolution conv = new convolution("numbers/"+helper.Last(), int.Parse(helper[helper.Length - 2]), network);
            }
        }
    }
}